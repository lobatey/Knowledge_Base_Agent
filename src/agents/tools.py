import json

from langchain_core.tools import BaseTool, tool

from src.agents.rag_retriever import HybridRAGRetriever, format_docs_with_sources


_retriever: HybridRAGRetriever | None = None


def get_retriever() -> HybridRAGRetriever:
    global _retriever

    if _retriever is None:
        _retriever = HybridRAGRetriever(
            persist_directory="./chroma_db",
            vector_k=10,
            bm25_k=10,
            final_k=5,
        )

    return _retriever


def refresh_retriever() -> None:
    """
    Call this after adding, deleting, or updating documents.
    """
    global _retriever

    if _retriever is not None:
        _retriever.refresh_bm25_index()


def database_search_func(query: str, source: str | None = None) -> str:
    """
    Search the knowledge base using hybrid retrieval.

    Args:
        query: The user question or rewritten search query.
        source: Optional source filename. If provided, search only within this document.
    """

    retriever = get_retriever()

    metadata_filter = None

    if source and source.strip():
        metadata_filter = {
            "source": source.strip(),
        }

    docs = retriever.retrieve(
        query=query,
        metadata_filter=metadata_filter,
    )

    if not docs:
        if metadata_filter:
            return (
                "检索状态：NO_EVIDENCE\n"
                f"检索范围：仅文档 {source}\n"
                "说明：在指定文档中没有检索到能够回答问题的知识库内容。\n"
                "回答要求：请明确回答“该文档中没有找到足够依据”，不要使用其他文档或模型常识补充。"
            )

        return (
            "检索状态：NO_EVIDENCE\n"
            "检索范围：全部知识库\n"
            "说明：知识库中没有检索到能够回答问题的相关内容。\n"
            "回答要求：请明确回答“知识库中没有找到足够依据”，不要使用模型常识编造答案。"
        )

    return (
            "检索状态：HAS_EVIDENCE\n"
            "说明：以下内容是知识库中检索到的依据。最终回答必须只基于这些依据。\n\n"
            + format_docs_with_sources(docs)
    )

def list_documents_func() -> str:
    """List all documents currently stored in the knowledge base."""

    retriever = HybridRAGRetriever(persist_directory="./chroma_db")
    data = retriever.chroma.get()

    metadatas = data.get("metadatas", [])

    docs = {}

    for metadata in metadatas:
        source = metadata.get("source", "unknown")
        if source not in docs:
            docs[source] = {
                "source": source,
                "chunk_count": 0,
                "file_type": metadata.get("file_type", "unknown"),
            }
        docs[source]["chunk_count"] += 1

    return json.dumps(list(docs.values()), ensure_ascii=False, indent=2)

def delete_document_func(source: str) -> str:
    """Delete a document from the knowledge base by source filename."""

    retriever = HybridRAGRetriever(persist_directory="./chroma_db")

    data = retriever.chroma.get(where={"source": source})
    ids = data.get("ids", [])

    if not ids:
        return f"没有找到 source={source} 的文档。"

    retriever.chroma.delete(ids=ids)

    return f"已删除文档 {source}，共删除 {len(ids)} 个 Chunk。"


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # Update name with the purpose of your database

list_documents: BaseTool = tool(list_documents_func)
list_documents.name = "List_Documents"

delete_document: BaseTool = tool(delete_document_func)
delete_document.name = "Delete_Document"

tools = [database_search, list_documents]
