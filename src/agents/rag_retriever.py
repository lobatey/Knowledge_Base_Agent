from __future__ import annotations

import os
import jieba
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

load_dotenv()


class HybridRAGRetriever:
    """
    Hybrid RAG Retriever:
    1. Vector Search: ChromaDB semantic search
    2. BM25 Search: keyword search
    3. Merge and deduplicate results
    4. Return final Top-K documents
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        vector_k: int = 10,
        bm25_k: int = 10,
        final_k: int = 5,
    ):
        self.persist_directory = persist_directory
        self.vector_k = vector_k
        self.bm25_k = bm25_k
        self.final_k = final_k

        self.embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-v1"),
        api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("COMPATIBLE_API_KEY"),
        base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("COMPATIBLE_BASE_URL"),
        check_embedding_ctx_length=False,
    )

        self.chroma = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

        # BM25 缓存：初始化时构建一次
        self._bm25_docs: list[Document] = []
        self._bm25_tokenized_corpus: list[list[str]] = []
        self._bm25_index: BM25Okapi | None = None

        self.refresh_bm25_index()

    # 给 BM25 用的。BM25需要完整文本语料，
    # 所以要从 Chroma 里把 chunk 取出来
    def _load_documents_from_chroma(
        self,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Load documents from ChromaDB for BM25 keyword retrieval.
        """

        if metadata_filter:
            data = self.chroma.get(where=metadata_filter)
        else:
            data = self.chroma.get()

        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        docs: list[Document] = []

        for content, metadata in zip(documents, metadatas, strict=False):
            if not content:
                continue

            docs.append(
                Document(
                    page_content=content,
                    metadata=metadata or {},
                )
            )

        return docs

    def refresh_bm25_index(
            self,
            metadata_filter: dict[str, Any] | None = None,
    ) -> None:
        """
        Build or rebuild BM25 index from ChromaDB documents.

        This method should be called:
        1. when the retriever is initialized;
        2. after adding documents;
        3. after deleting documents;
        4. after updating documents.
        """

        self._bm25_docs = self._load_documents_from_chroma(
            metadata_filter=metadata_filter,
        )

        if not self._bm25_docs:
            self._bm25_tokenized_corpus = []
            self._bm25_index = None
            return

        self._bm25_tokenized_corpus = [
            self._tokenize(doc.page_content)
            for doc in self._bm25_docs
        ]

        self._bm25_index = BM25Okapi(self._bm25_tokenized_corpus)

    # 生成文档唯一标识，用于去重
    def _document_key(self, doc: Document) -> str:
        """
        Generate a stable key for deduplication.
        """

        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        chunk_id = doc.metadata.get("chunk_id", "")

        if chunk_id:
            return str(chunk_id)

        return f"{source}:{page}:{hash(doc.page_content)}"

    # 把文本切成token，BM25使用
    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenizer for BM25.

        This is enough for English.
        For Chinese enterprise documents, you should replace this with jieba.
        """

        return list(jieba.cut(text.lower()))

    # 向量检索 用户问题 → embedding → ChromaDB 相似度搜索
    def _vector_search(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        search_kwargs: dict[str, Any] = {
            "k": self.vector_k,
        }

        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        retriever = self.chroma.as_retriever(
            search_kwargs=search_kwargs,
        )

        return retriever.invoke(query)

    # 关键词检索
    def _bm25_search(
            self,
            query: str,
            metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        BM25 keyword search using cached BM25 index.
        """

        # 如果有 metadata_filter，说明这次查询只查部分文档。
        # 简单处理方式：临时为过滤后的文档构建 BM25，不污染全局缓存。
        if metadata_filter:
            filtered_docs = self._load_documents_from_chroma(
                metadata_filter=metadata_filter,
            )

            if not filtered_docs:
                return []

            tokenized_corpus = [
                self._tokenize(doc.page_content)
                for doc in filtered_docs
            ]

            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = self._tokenize(query)
            scores = bm25.get_scores(tokenized_query)

            ranked_docs = sorted(
                zip(filtered_docs, scores, strict=False),
                key=lambda item: item[1],
                reverse=True,
            )

            return [
                doc
                for doc, score in ranked_docs[: self.bm25_k]
                if score > 0
            ]

        # 没有 metadata_filter 时，直接使用缓存好的 BM25 索引
        if self._bm25_index is None or not self._bm25_docs:
            return []

        tokenized_query = self._tokenize(query)

        scores = self._bm25_index.get_scores(tokenized_query)

        ranked_docs = sorted(
            zip(self._bm25_docs, scores, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            doc
            for doc, score in ranked_docs[: self.bm25_k]
            if score > 0
        ]

    # 智能地结合向量检索（语义）和 BM25（关键词）的排序信息，
    # 让两个列表都“投票”的文档排在前面，通常能显著提升混合检索质量
    def _rrf_fusion(
            self,
            vector_docs: list[Document],
            bm25_docs: list[Document],
            k: int = 60,
    ) -> list[Document]:
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(vector_docs, start=1):
            key = self._document_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs, start=1):
            key = self._document_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

        ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results = []
        for key in ranked_keys:
            doc = doc_map[key]
            doc.metadata["hybrid_score"] = scores[key]
            results.append(doc)

        return results

    def retrieve(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Execute hybrid retrieval:
        - Vector Search from ChromaDB
        - BM25 Search from all ChromaDB documents
        - Merge and deduplicate
        """

        vector_docs = self._vector_search(
            query=query,
            metadata_filter=metadata_filter,
        )

        bm25_docs = self._bm25_search(
            query=query,
            metadata_filter=metadata_filter,
        )

        fused_docs = self._rrf_fusion(vector_docs, bm25_docs)

        return fused_docs[: self.final_k]

    def retrieve_debug(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Debug hybrid retrieval process.

        This method is used for RAG debugging:
        1. show vector search results;
        2. show BM25 keyword search results;
        3. show final RRF fused results.

        It does not call LLM.
        """

        vector_docs = self._vector_search(
            query=query,
            metadata_filter=metadata_filter,
        )

        bm25_docs = self._bm25_search(
            query=query,
            metadata_filter=metadata_filter,
        )

        fused_docs = self._rrf_fusion(vector_docs, bm25_docs)
        final_docs = fused_docs[: self.final_k]

        def serialize_doc(doc: Document, rank: int) -> dict[str, Any]:
            content = doc.page_content or ""

            return {
                "rank": rank,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "file_type": doc.metadata.get("file_type", "unknown"),
                "hybrid_score": doc.metadata.get("hybrid_score"),
                "content": content,
                "preview": content[:300],
                "metadata": doc.metadata,
            }

        return {
            "query": query,
            "metadata_filter": metadata_filter,
            "vector_results": [
                serialize_doc(doc, index)
                for index, doc in enumerate(vector_docs, start=1)
            ],
            "bm25_results": [
                serialize_doc(doc, index)
                for index, doc in enumerate(bm25_docs, start=1)
            ],
            "final_results": [
                serialize_doc(doc, index)
                for index, doc in enumerate(final_docs, start=1)
            ],
            "stats": {
                "vector_count": len(vector_docs),
                "bm25_count": len(bm25_docs),
                "final_count": len(final_docs),
            },
        }


# 格式化检索结果 把检索到的文档转换成适合塞进 Prompt 的上下文
def format_docs_with_sources(docs) -> str:
    """
    Format retrieved documents with explicit citation markers.

    The returned text will be used as tool output for the LLM.
    Each chunk is assigned a citation id like [1], [2], [3].
    """

    if not docs:
        return "没有检索到相关知识库内容。"

    formatted_chunks = []

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}

        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        chunk_id = metadata.get("chunk_id", "unknown")
        file_type = metadata.get("file_type", "unknown")
        hybrid_score = metadata.get("hybrid_score")

        content = doc.page_content or ""
        content = content.strip()

        score_text = ""
        if hybrid_score is not None:
            score_text = f"\n相似度/融合分数：{hybrid_score}"

        formatted_chunks.append(
            f"""[引用 {index}]
                来源文档：{source}
                页码：{page}
                Chunk ID：{chunk_id}
                文件类型：{file_type}{score_text}
                内容：
                {content}
                """
        )

    source_summary = []

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}

        source_summary.append(
            {
                "index": index,
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", "unknown"),
                "chunk_id": metadata.get("chunk_id", "unknown"),
            }
        )

    source_lines = [
        f"[{item['index']}] {item['source']} ｜ page: {item['page']} ｜ chunk_id: {item['chunk_id']}"
        for item in source_summary
    ]

    return (
        "以下是从知识库检索到的相关内容。回答时必须基于这些内容，并在关键结论后标注引用编号，例如：[1]、[2]。\n\n"
        + "\n---\n".join(formatted_chunks)
        + "\n\n可用引用来源列表：\n"
        + "\n".join(source_lines)
    )



if __name__ == "__main__":
    retriever = HybridRAGRetriever(
        persist_directory="./chroma_db",
        vector_k=10,
        bm25_k=10,
        final_k=5,
    )

    query = "What's my company's mission and values"

    docs = retriever.retrieve(
        query=query,
        metadata_filter=None,
        # 示例：
        # metadata_filter={"department": "general"},
        # metadata_filter={"kb_type": "enterprise"},
        # metadata_filter={"source": "employee_handbook.pdf"},
    )

    print(format_docs_with_sources(docs))