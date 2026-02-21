import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)

_TRACE_SESSION: ContextVar["DebugTraceSession | None"] = ContextVar(
    "debug_trace_session",
    default=None,
)

_LOGGING_CONFIGURED = False
_LOGGING_SIGNATURE: tuple[str, str, str] | None = None


def _safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Document):
        return {
            "type": "Document",
            "page_content": value.page_content,
            "metadata": _to_serializable(value.metadata),
        }
    if isinstance(value, BaseModel):
        return _to_serializable(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_to_serializable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Exception):
        return {"type": value.__class__.__name__, "message": str(value)}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    keys = set(before) | set(after)
    for key in sorted(keys):
        before_value = before.get(key, "__missing__")
        after_value = after.get(key, "__missing__")
        if before_value != after_value:
            delta[key] = {
                "before": before_value,
                "after": after_value,
            }
    return delta


def configure_logging(app_name: str = "rag") -> None:
    """
    Configure root logging with console + file handlers.

    This is idempotent and safe to call multiple times.
    """
    global _LOGGING_CONFIGURED
    global _LOGGING_SIGNATURE

    level_name = settings.debug.log_level.upper()
    signature = (settings.debug.log_dir, level_name, app_name)
    if _LOGGING_CONFIGURED and signature == _LOGGING_SIGNATURE:
        return

    log_level = getattr(logging, level_name, logging.INFO)

    log_dir = Path(settings.debug.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_dir / f"{app_name}.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Keep noisy HTTP logs under control.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True
    _LOGGING_SIGNATURE = signature


def get_active_trace_session() -> "DebugTraceSession | None":
    return _TRACE_SESSION.get()


def log_trace_event(event_type: str, payload: dict[str, Any]) -> None:
    session = get_active_trace_session()
    if session is None:
        return
    session.log_event(event_type=event_type, payload=payload)


def merge_state(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    merged.update(update)
    return merged


class DebugTraceSession:
    def __init__(self, run_id: str, query: str, mode: str) -> None:
        self.run_id = run_id
        self.query = query
        self.mode = mode
        self.started_at = time.perf_counter()
        self.events = 0

        self.log_dir = Path(settings.debug.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.trace_path = self.log_dir / f"trace-{stamp}-{run_id}.jsonl"

    @classmethod
    def create(cls, query: str, mode: str) -> "DebugTraceSession":
        run_id = uuid.uuid4().hex
        return cls(run_id=run_id, query=query, mode=mode)

    def _write(self, record: dict[str, Any]) -> None:
        serialized = _safe_json(record)
        with self.trace_path.open("a", encoding="utf-8") as file:
            file.write(serialized)
            file.write(os.linesep)

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events += 1
        self._write(
            {
                "ts": _utc_now_iso(),
                "run_id": self.run_id,
                "sequence": self.events,
                "event_type": event_type,
                "payload": _to_serializable(payload),
            }
        )

    def record_run_start(self, initial_state: dict[str, Any]) -> None:
        self.log_event(
            "run_start",
            {
                "mode": self.mode,
                "query": self.query,
                "initial_state": initial_state,
            },
        )

    def record_node_start(
        self,
        node_name: str,
        state_before: dict[str, Any],
    ) -> None:
        self.log_event(
            "node_start",
            {
                "node": node_name,
                "state_before": state_before,
            },
        )

    def record_node_end(
        self,
        node_name: str,
        state_before: dict[str, Any],
        node_output: dict[str, Any],
        state_after: dict[str, Any],
        duration_ms: float,
    ) -> None:
        self.log_event(
            "node_end",
            {
                "node": node_name,
                "duration_ms": round(duration_ms, 3),
                "node_output": node_output,
                "state_after": state_after,
                "state_delta": _state_delta(state_before, state_after),
            },
        )

    def record_node_error(
        self,
        node_name: str,
        state_before: dict[str, Any],
        error: Exception,
    ) -> None:
        self.log_event(
            "node_error",
            {
                "node": node_name,
                "state_before": state_before,
                "error": error,
            },
        )

    def record_run_end(self, final_state: dict[str, Any]) -> None:
        self.log_event(
            "run_end",
            {
                "duration_ms": round((time.perf_counter() - self.started_at) * 1000, 3),
                "final_state": final_state,
            },
        )

    def record_run_exception(self, error: Exception) -> None:
        self.log_event(
            "run_exception",
            {
                "duration_ms": round((time.perf_counter() - self.started_at) * 1000, 3),
                "error": error,
            },
        )


@contextmanager
def debug_run(
    query: str,
    initial_state: dict[str, Any],
    mode: str = "invoke",
):
    """
    Context manager for full run-level tracing.

    Yields the active `DebugTraceSession` when enabled, otherwise `None`.
    """
    configure_logging()
    if not settings.debug.enabled:
        yield None
        return

    session = DebugTraceSession.create(query=query, mode=mode)
    token: Token = _TRACE_SESSION.set(session)
    session.record_run_start(initial_state=initial_state)

    try:
        yield session
    except Exception as error:
        session.record_run_exception(error)
        raise
    finally:
        _TRACE_SESSION.reset(token)
