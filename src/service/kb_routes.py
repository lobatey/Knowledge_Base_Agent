import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from src.kb.document_service import (
    add_document,
    add_document_with_task,
    list_documents,
    list_document_chunks,
    delete_document,
)
from src.kb.task_store import create_task, get_task
from src.agents.rag_retriever import HybridRAGRetriever

router = APIRouter(prefix="/kb", tags=["Knowledge Base"])


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    suffix = Path(file.filename).suffix.lower()

    if suffix not in [".pdf", ".docx", ".txt", ".md"]:
        raise HTTPException(
            status_code=400,
            detail="不支持的文件类型，仅支持 pdf、docx、txt、md",
        )

    try:
        upload_dir = Path("data")
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = Path(file.filename).name
        target_path = upload_dir / safe_filename

        print("[KB ROUTE] file.filename =", file.filename, flush=True)
        print("[KB ROUTE] safe_filename =", safe_filename, flush=True)

        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        task_id = create_task(safe_filename)

        background_tasks.add_task(
            add_document_with_task,
            target_path,
            task_id,
        )

        return {
            "success": True,
            "message": "文档已上传，正在后台入库",
            "task_id": task_id,
        }

    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

@router.get("/tasks/{task_id}")
async def get_upload_task(task_id: str):
    task = get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    return {
        "success": True,
        "data": task,
    }


@router.get("/documents")
async def get_documents():
    return {
        "success": True,
        "data": list_documents(),
    }

@router.get("/documents/{source}/chunks")
async def get_document_chunks(source: str):
    """
    Get all chunks of a specific document.

    This API is used for inspecting how a document is split and stored in ChromaDB.
    """

    if not source or not source.strip():
        raise HTTPException(status_code=400, detail="source 不能为空")

    try:
        chunks = list_document_chunks(source)

        return {
            "success": True,
            "data": {
                "source": source,
                "total": len(chunks),
                "chunks": chunks,
            },
        }

    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@router.delete("/documents/{source}")
async def remove_document(source: str):
    return {
        "success": True,
        "data": delete_document(source),
    }

@router.get("/debug/retrieve")
async def debug_retrieve(
    query: str,
    source: str | None = None,
):
    """
    Debug knowledge base retrieval.

    This API only performs retrieval and does not call LLM.
    It is useful for checking whether the RAG system retrieves the right chunks.
    """

    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    try:
        metadata_filter = None

        if source and source.strip():
            metadata_filter = {
                "source": source.strip(),
            }

        retriever = HybridRAGRetriever(
            persist_directory="./chroma_db",
            vector_k=10,
            bm25_k=10,
            final_k=5,
        )

        result = retriever.retrieve_debug(
            query=query.strip(),
            metadata_filter=metadata_filter,
        )

        return {
            "success": True,
            "data": result,
        }

    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
