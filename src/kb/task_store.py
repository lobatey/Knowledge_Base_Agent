from datetime import datetime
from typing import Any, Dict
from uuid import uuid4


TASKS: Dict[str, Dict[str, Any]] = {}


def create_task(filename: str) -> str:
    task_id = str(uuid4())

    TASKS[task_id] = {
        "task_id": task_id,
        "filename": filename,
        "status": "pending",
        "message": "等待处理",
        "progress": 0,
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    return task_id


def update_task(
    task_id: str,
    status: str | None = None,
    message: str | None = None,
    progress: int | None = None,
    result: Any | None = None,
    error: str | None = None,
) -> None:
    if task_id not in TASKS:
        return

    if status is not None:
        TASKS[task_id]["status"] = status

    if message is not None:
        TASKS[task_id]["message"] = message

    if progress is not None:
        TASKS[task_id]["progress"] = progress

    if result is not None:
        TASKS[task_id]["result"] = result

    if error is not None:
        TASKS[task_id]["error"] = error

    TASKS[task_id]["updated_at"] = datetime.now().isoformat()


def get_task(task_id: str) -> Dict[str, Any] | None:
    return TASKS.get(task_id)
