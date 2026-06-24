from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.settings import settings
from src.memory.long_term_memory import (
    create_memory,
    delete_memory,
    format_memories_for_prompt,
    get_memory_db_path,
    get_memory_stats,
    list_memories,
)


router = APIRouter(prefix="/memory", tags=["Memory"])


class CreateMemoryRequest(BaseModel):
    user_id: str = Field(default="default", description="User id")
    memory_type: Literal["preference", "profile", "project", "note"] = "preference"
    content: str = Field(..., min_length=1, description="Memory content")


@router.get("/status")
async def memory_status(user_id: str = "default"):
    """
    Check memory status.

    Short-term memory is provided by LangGraph checkpointer.
    Long-term memory is provided by local SQLite table.
    """

    stats = get_memory_stats(user_id=user_id)

    return {
        "success": True,
        "data": {
            "short_term_memory": {
                "enabled": True,
                "backend": settings.DATABASE_TYPE,
                "sqlite_checkpoint_path": settings.SQLITE_DB_PATH,
                "description": "LangGraph checkpointer stores thread-scoped conversation state.",
            },
            "long_term_memory": {
                "enabled": True,
                "backend": "sqlite",
                "db_path": get_memory_db_path(),
                "memory_count": stats["memory_count"],
                "description": "Long-term memory stores user-level persistent facts and preferences.",
            },
        },
    }


@router.post("/memories")
async def add_memory(request: CreateMemoryRequest):
    """
    Create a long-term memory item.
    """

    try:
        memory = create_memory(
            user_id=request.user_id,
            memory_type=request.memory_type,
            content=request.content,
        )

        return {
            "success": True,
            "data": memory,
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}")


@router.get("/memories")
async def get_memories(user_id: str = "default"):
    """
    List long-term memories.
    """

    return {
        "success": True,
        "data": list_memories(user_id=user_id),
    }


@router.delete("/memories/{memory_id}")
async def remove_memory(memory_id: int, user_id: str = "default"):
    """
    Delete a long-term memory item.
    """

    deleted = delete_memory(
        memory_id=memory_id,
        user_id=user_id,
    )

    if not deleted:
        raise HTTPException(status_code=404, detail="memory not found")

    return {
        "success": True,
        "data": {
            "deleted": True,
            "memory_id": memory_id,
        },
    }


@router.get("/context")
async def get_memory_context(
    user_id: str = "default",
    limit: int = 8,
):
    """
    Return formatted long-term memory context for prompt injection.
    """

    context = format_memories_for_prompt(
        user_id=user_id,
        limit=limit,
    )

    return {
        "success": True,
        "data": {
            "user_id": user_id,
            "context": context,
        },
    }
