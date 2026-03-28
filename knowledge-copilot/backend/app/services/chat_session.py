import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from app.core.config import settings

# ── Session store (in-memory, persisted to disk) ──────────────────────────────

_sessions: Dict[str, dict] = {}
_SESSION_DIR = Path(settings.upload_dir).parent / "sessions"


def _session_path(session_id: str) -> Path:
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSION_DIR / f"{session_id}.json"


def create_session() -> str:
    """Create a new chat session. Returns the session ID."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "id":         session_id,
        "created_at": datetime.utcnow().isoformat(),
        "messages":   [],
    }
    _persist(session_id)
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    """Return session dict or None if not found."""
    # Check memory first, then disk
    if session_id in _sessions:
        return _sessions[session_id]

    path = _session_path(session_id)
    if path.exists():
        with open(path) as f:
            session = json.load(f)
        _sessions[session_id] = session
        return session

    return None


def add_message(session_id: str, role: str, content: str):
    """Append a message to the session history."""
    session = get_session(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    session["messages"].append({
        "role":      role,
        "content":   content,
        "timestamp": datetime.utcnow().isoformat(),
    })
    _persist(session_id)


def get_history(session_id: str) -> List[dict]:
    """Return the message history for a session (role + content only)."""
    session = get_session(session_id)
    if not session:
        return []
    return [
        {"role": m["role"], "content": m["content"]}
        for m in session["messages"]
    ]


def list_sessions() -> List[dict]:
    """Return metadata for all sessions."""
    # Load any sessions from disk not yet in memory
    if _SESSION_DIR.exists():
        for path in _SESSION_DIR.glob("*.json"):
            sid = path.stem
            if sid not in _sessions:
                get_session(sid)

    return [
        {
            "id":           s["id"],
            "created_at":   s["created_at"],
            "message_count": len(s["messages"]),
        }
        for s in _sessions.values()
    ]


def delete_session(session_id: str) -> bool:
    """Delete a session from memory and disk."""
    _sessions.pop(session_id, None)
    path = _session_path(session_id)
    if path.exists():
        path.unlink()
        return True
    return False


def _persist(session_id: str):
    """Write session to disk as JSON."""
    session = _sessions.get(session_id)
    if session:
        with open(_session_path(session_id), "w") as f:
            json.dump(session, f, indent=2)