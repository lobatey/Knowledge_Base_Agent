import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
import traceback

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.kb.task_store import update_task
from src.kb.document_processor import (
    SUPPORTED_EXTENSIONS,
    extract_documents,
    smart_split_documents,
    document_type,
)

try:
    from src.agents.rag_retriever import refresh_retriever
except Exception:
    refresh_retriever = None


DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "markdown",
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)



def load_and_split_document(file_path: Path) -> List:
    """
    文档解析与切分入口。

    新流程：
    1. extract_documents：解析 PDF / DOCX / TXT / MD
    2. 清洗页眉、页脚、页码、乱码控制字符
    3. 表格转 Markdown
    4. 图片 OCR
    5. 公式单独保留
    6. smart_split_documents：智能切分
    7. 统一补充 ChromaDB metadata
    """

    documents = extract_documents(file_path)
    chunks = smart_split_documents(documents)

    file_type = document_type(file_path)

    cleaned_chunks = []

    for index, chunk in enumerate(chunks):
        text = (chunk.page_content or "").strip()

        if not text:
            continue

        content_type = chunk.metadata.get("content_type", "text")

        caption_label = chunk.metadata.get("caption_label")

        if caption_label:
            safe_caption_label = (
                str(caption_label)
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
                .replace("：", "_")
                .replace(":", "_")
            )
            chunk_id = f"{file_path.stem}_{content_type}_{safe_caption_label}_chunk_{index}"
        else:
            chunk_id = f"{file_path.stem}_{content_type}_chunk_{index}"

        chunk.metadata.update(
            {
                "source": file_path.name,
                "chunk_id": chunk_id,
                "file_type": file_type,
                "content_type": content_type,
            }
        )

        cleaned_chunks.append(chunk)

    return cleaned_chunks


def get_vector_store() -> Chroma:
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("COMPATIBLE_BASE_URL")
    embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("COMPATIBLE_API_KEY")

    print("[KB] EMBEDDING_MODEL =", embedding_model, flush=True)
    print("[KB] EMBEDDING_BASE_URL =", embedding_base_url, flush=True)
    print(
        "[KB] EMBEDDING_API_KEY_HEAD =",
        embedding_api_key[:6] if embedding_api_key else None,
        flush=True,
    )

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=embedding_api_key,
        base_url=embedding_base_url,
        check_embedding_ctx_length=False,
        timeout=180,
        max_retries=1,
    )

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def add_document(file_path: Path) -> Dict[str, Any]:
    ensure_dirs()
    print("[KB] 开始入库:", file_path, flush=True)

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"不支持的文件类型，仅支持: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )

    target_path = DATA_DIR / file_path.name

    if file_path.resolve() != target_path.resolve():
        shutil.copyfile(file_path, target_path)

    print("[KB] 文件已保存:", target_path, flush=True)

    chunks = load_and_split_document(target_path)
    print("[KB] 文档切分完成，chunks =", len(chunks), flush=True)

    if not chunks:
        raise ValueError(
            "文档没有解析出有效内容。可能是扫描版 PDF 且未启用 OCR，或者文件内容为空。"
        )

    vector_store = get_vector_store()
    print("[KB] Chroma 已加载", flush=True)

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    print("[KB] 开始写入 Chroma / 调用 Embedding", flush=True)

    vector_store.add_documents(chunks, ids=ids)
    print("[KB] 写入 Chroma 完成", flush=True)

    if refresh_retriever:
        print("[KB] 开始刷新 retriever", flush=True)
        refresh_retriever()
        print("[KB] 刷新 retriever 完成", flush=True)

    return {
        "source": target_path.name,
        "chunks": len(chunks),
        "file_type": SUPPORTED_EXTENSIONS[target_path.suffix.lower()],
    }



def list_documents() -> List[Dict[str, Any]]:
    ensure_dirs()

    vector_store = get_vector_store()
    collection = vector_store.get()

    metadatas = collection.get("metadatas", [])

    documents = {}

    for metadata in metadatas:
        if not metadata:
            continue

        source = metadata.get("source", "unknown")

        if source not in documents:
            documents[source] = {
                "source": source,
                "file_type": metadata.get("file_type", "unknown"),
                "chunks": 0,
                "content_types": {},
            }

        documents[source]["chunks"] += 1
        content_type = metadata.get("content_type", "text")
        documents[source]["content_types"][content_type] = (
                documents[source]["content_types"].get(content_type, 0) + 1
        )

    # return list(documents.values())
    return sorted(
        documents.values(),
        key=lambda item: item.get("source", ""),
    )

# 从 ChromaDB 中读取某个文档的所有 Chunk
def list_document_chunks(source: str) -> List[Dict[str, Any]]:
    """
    List all chunks of a specific document from ChromaDB.

    Args:
        source: The source filename stored in chunk metadata.

    Returns:
        A list of chunks with content and metadata.
    """

    ensure_dirs()

    if not source or not source.strip():
        return []

    vector_store = get_vector_store()
    collection = vector_store.get(
        where={
            "source": source.strip(),
        }
    )

    documents = collection.get("documents", [])
    metadatas = collection.get("metadatas", [])
    ids = collection.get("ids", [])

    chunks: list[dict[str, Any]] = []

    for index, content in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        chunk_id = ids[index] if index < len(ids) else metadata.get("chunk_id", f"{source}_chunk_{index}")

        text = content or ""

        chunks.append(
            {
                "index": index,
                "chunk_id": metadata.get("chunk_id", chunk_id),
                "source": metadata.get("source", source),
                "page": metadata.get("page", "unknown"),
                "file_type": metadata.get("file_type", "unknown"),
                "content_type": metadata.get("content_type", "text"),
                "caption": metadata.get("caption", ""),
                "caption_type": metadata.get("caption_type", ""),
                "caption_label": metadata.get("caption_label", ""),
                "content": text,
                "preview": text[:300],
                "metadata": metadata,
            }
        )

    chunks.sort(
        key=lambda item: (
            item.get("page", 0) if isinstance(item.get("page", 0), int) else 0,
            item.get("index", 0),
        )
    )

    return chunks


def delete_document(source: str) -> Dict[str, Any]:
    ensure_dirs()

    vector_store = get_vector_store()
    collection = vector_store.get()

    ids_to_delete = []

    for item_id, metadata in zip(
        collection.get("ids", []),
        collection.get("metadatas", []),
    ):
        if metadata and metadata.get("source") == source:
            ids_to_delete.append(item_id)

    if not ids_to_delete:
        return {
            "source": source,
            "deleted_chunks": 0,
            "message": "未找到该文档",
        }

    vector_store.delete(ids=ids_to_delete)

    file_path = DATA_DIR / source
    if file_path.exists():
        file_path.unlink()

    if refresh_retriever:
        refresh_retriever()

    return {
        "source": source,
        "deleted_chunks": len(ids_to_delete),
        "message": "删除成功",
    }

def add_document_with_task(file_path: Path, task_id: str):
    try:
        update_task(task_id, status="running", message="开始处理文档", progress=10)

        ensure_dirs()

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件类型，仅支持: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
            )

        target_path = DATA_DIR / file_path.name

        if file_path.resolve() != target_path.resolve():
            shutil.copyfile(file_path, target_path)

        update_task(task_id, message="文档已保存，开始解析", progress=25)

        chunks = load_and_split_document(target_path)

        if not chunks:
            raise ValueError(
                "文档没有解析出有效内容。可能是扫描版 PDF 且未启用 OCR，或者文件内容为空。"
            )

        content_type_stats = {}

        for chunk in chunks:
            content_type = chunk.metadata.get("content_type", "text")
            content_type_stats[content_type] = content_type_stats.get(content_type, 0) + 1

        stats_text = "，".join(
            f"{key}:{value}"
            for key, value in sorted(content_type_stats.items())
        )

        update_task(
            task_id,
            message=f"文档解析完成，共 {len(chunks)} 个分块（{stats_text}），开始向量化",
            progress=50,
        )

        vector_store = get_vector_store()
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]

        vector_store.add_documents(chunks, ids=ids)

        update_task(task_id, message="向量写入完成，刷新检索器", progress=85)

        if refresh_retriever:
            refresh_retriever()

        result = {
            "source": target_path.name,
            "chunks": len(chunks),
            "file_type": SUPPORTED_EXTENSIONS[target_path.suffix.lower()],
        }

        update_task(
            task_id,
            status="success",
            message="文档入库完成",
            progress=100,
            result=result,
        )

    except Exception as exc:
        traceback.print_exc()
        update_task(
            task_id,
            status="failed",
            message="文档入库失败",
            progress=100,
            error=f"{type(exc).__name__}: {exc}",
        )