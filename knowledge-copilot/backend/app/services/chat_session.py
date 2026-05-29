import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Session store (in-memory, persisted to disk) ──────────────────────────────

_sessions: Dict[str, dict] = {}
_SESSION_DIR = Path("data") / "sessions"


def _session_path(session_id: str) -> Path:
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSION_DIR / f"{session_id}.json"


def create_session() -> str:
    """Create a new chat session. Returns the session ID."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "id":         session_id,
        "title":      "New conversation",
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
        session.setdefault("title", _derive_title(session))
        _sessions[session_id] = session
        return session

    return None


def _derive_title(session: dict) -> str:
    """Derive a title from the first user message, or return a default."""
    for m in session.get("messages", []):
        if m["role"] == "user":
            content = m["content"][:80]
            return content + "…" if len(m["content"]) > 80 else content
    return "New conversation"


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

    if role == "user" and session.get("title", "New conversation") == "New conversation":
        session["title"] = content[:80]

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
            "title":        s.get("title", "New conversation"),
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


def rename_session(session_id: str, new_title: str) -> Optional[dict]:
    """Rename a session. Returns the updated session or None if not found."""
    session = get_session(session_id)
    if not session:
        return None
    session["title"] = new_title.strip()[:200] or "New conversation"
    _persist(session_id)
    return session


def _persist(session_id: str):
    """Write session to disk as JSON."""
    session = _sessions.get(session_id)
    if session:
        with open(_session_path(session_id), "w") as f:
            json.dump(session, f, indent=2)