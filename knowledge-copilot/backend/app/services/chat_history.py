import logging
from datetime import datetime
from typing import Optional
from bson import ObjectId
from app.models.database import get_db
from app.core.config import settings

logger = logging.getLogger("knowledge_copilot.chat_history")


# ── Sessions ──────────────────────────────────────────────────────────────────

async def create_session(user_id: str, title: str = "New conversation") -> str:
    """
    Create a new chat session for a user.
    Returns the session_id as a string.
    """
    db  = get_db()
    now = datetime.utcnow()

    result = await db.chat_sessions.insert_one({
        "user_id":    user_id,
        "title":      title,
        "created_at": now,
        "updated_at": now,
        "is_active":  True,
    })
    return str(result.inserted_id)


async def get_session(session_id: str, user_id: str) -> Optional[dict]:
    """
    Fetch a session — enforces ownership (user_id must match).
    Returns None if not found or not owned by user.
    """
    db = get_db()
    if not ObjectId.is_valid(session_id):
        return None

    session = await db.chat_sessions.find_one({
        "_id":     ObjectId(session_id),
        "user_id": user_id,           # ownership check
        "is_active": True,
    })
    if session:
        session["id"] = str(session["_id"])
    return session


async def list_sessions(user_id: str, limit: int = 20) -> list[dict]:
    """
    Return a user's chat sessions, most recent first.
    Lightweight — only returns metadata, not messages.
    """
    db = get_db()
    cursor = db.chat_sessions.find(
        {"user_id": user_id, "is_active": True},
        sort=[("updated_at", -1)],
        limit=limit,
    )
    sessions = []
    async for s in cursor:
        s["id"] = str(s["_id"])
        message_count = await db.chat_messages.count_documents(
            {"session_id": s["id"]}
        )
        sessions.append({
            "id":            s["id"],
            "title":         s["title"],
            "created_at":    s["created_at"],
            "updated_at":    s["updated_at"],
            "message_count": message_count,
        })
    return sessions


async def delete_session(session_id: str, user_id: str) -> bool:
    """
    Soft-delete a session (sets is_active=False).
    Enforces ownership — returns False if not found or not owned.
    """
    db = get_db()
    if not ObjectId.is_valid(session_id):
        return False

    result = await db.chat_sessions.update_one(
        {"_id": ObjectId(session_id), "user_id": user_id},
        {"$set": {"is_active": False, "updated_at": datetime.utcnow()}},
    )
    return result.modified_count > 0


async def rename_session(session_id: str, user_id: str, new_title: str) -> Optional[dict]:
    """
    Rename a session. Enforces ownership.
    Returns the updated session dict or None if not found.
    """
    db = get_db()
    if not ObjectId.is_valid(session_id):
        return None

    result = await db.chat_sessions.find_one_and_update(
        {"_id": ObjectId(session_id), "user_id": user_id, "is_active": True},
        {"$set": {"title": new_title.strip()[:200], "updated_at": datetime.utcnow()}},
        return_document=True,
    )
    if result:
        result["id"] = str(result["_id"])
    return result


async def _update_session_title(session_id: str, title: str):
    """Auto-update session title from first user message (truncated to 60 chars)."""
    db = get_db()
    await db.chat_sessions.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {
            "title":      title[:60],
            "updated_at": datetime.utcnow(),
        }},
    )


# ── Messages ──────────────────────────────────────────────────────────────────

async def save_user_message(
    session_id: str,
    user_id:    str,
    content:    str,
) -> str:
    """
    Save the user's question to the database.
    Also sets the session title from the first message.
    Returns the message _id.
    """
    db = get_db()

    result = await db.chat_messages.insert_one({
        "session_id":     session_id,
        "user_id":        user_id,
        "role":           "user",
        "content":        content,
        "sources":        [],
        "context_chunks": [],
        "model":          None,
        "created_at":     datetime.utcnow(),
    })

    # Set session title from first question if it's still the default
    session = await db.chat_sessions.find_one({"_id": ObjectId(session_id)})
    if session and session.get("title") in ("New conversation", ""):
        await _update_session_title(session_id, content)

    # Bump session.updated_at so it sorts to top of list
    await db.chat_sessions.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"updated_at": datetime.utcnow()}},
    )

    return str(result.inserted_id)


async def save_assistant_message(
    session_id:     str,
    user_id:        str,
    content:        str,
    sources:        list[dict],
    context_chunks: list[dict],
    model:          str,
) -> str:
    """
    Save the assistant's answer with full RAG context.

    sources        — the citations shown to the user (file, page, score)
    context_chunks — the full text chunks the LLM received (for auditing)
    model          — which LLM generated the answer

    Returns the message _id.
    """
    db = get_db()

    result = await db.chat_messages.insert_one({
        "session_id":     session_id,
        "user_id":        user_id,
        "role":           "assistant",
        "content":        content,
        "sources":        sources,
        "context_chunks": context_chunks,
        "model":          model,
        "created_at":     datetime.utcnow(),
    })

    # Bump session.updated_at so it sorts to top of list
    await db.chat_sessions.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"updated_at": datetime.utcnow()}},
    )

    return str(result.inserted_id)


async def get_messages(
    session_id: str,
    user_id:    str,
    limit:      int = 50,
) -> list[dict]:
    """
    Fetch all messages in a session, oldest first.
    Enforces ownership via user_id.
    """
    db = get_db()

    # Verify the session belongs to this user first
    session = await get_session(session_id, user_id)
    if not session:
        return []

    cursor = db.chat_messages.find(
        {"session_id": session_id},
        sort=[("created_at", 1)],
        limit=limit,
    )
    messages = []
    async for m in cursor:
        messages.append({
            "id":         str(m["_id"]),
            "role":       m["role"],
            "content":    m["content"],
            "sources":    m.get("sources", []),
            "model":      m.get("model"),
            "created_at": m["created_at"],
        })
    return messages


async def get_history_for_llm(
    session_id: str,
    user_id:    str,
    max_turns:  int = None,
) -> list[dict]:
    """
    Return relevant conversation history for the LLM.

    Uses the memory_manager for smarter history selection:
    - Dynamic windowing based on conversation complexity
    - Entity-aware relevance scoring (prefers turns mentioning same entities)
    - Conversation summarization for long chats (over compression threshold)

    Falls back to simple "last N turns" if memory_manager is unavailable.
    """
    max_turns = max_turns or settings.memory_max_turns
    db = get_db()

    # Fetch more turns than needed for the memory manager to select from
    fetch_limit = max(max_turns * 3, settings.memory_compression_threshold)
    cursor = db.chat_messages.find(
        {"session_id": session_id, "user_id": user_id},
        sort=[("created_at", -1)],
        limit=fetch_limit,
        projection={"role": 1, "content": 1},
    )
    messages = []
    async for m in cursor:
        messages.append({
            "role":    m["role"],
            "content": m["content"],
        })

    messages.reverse()

    if not messages:
        return messages

    # Use memory_manager for smarter selection
    try:
        from app.services.memory_manager import get_relevant_history, needs_compression
        if len(messages) > max_turns:
            selected = get_relevant_history(messages, messages[-1].get("content", ""), max_turns=max_turns)
            logger.info(
                f"Memory manager: selected {len(selected)}/{len(messages)} turns "
                f"(compression: {needs_compression(messages)})"
            )
            return selected
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Memory manager failed, using fallback: {e}")

    return messages[-max_turns:]


async def get_user_stats(user_id: str) -> dict:
    """
    Return aggregate stats for a user's chat history.
    Useful for a dashboard summary panel.
    """
    db = get_db()

    session_count = await db.chat_sessions.count_documents(
        {"user_id": user_id, "is_active": True}
    )
    message_count = await db.chat_messages.count_documents(
        {"user_id": user_id, "role": "user"}
    )

    # Most recent session
    last_session = await db.chat_sessions.find_one(
        {"user_id": user_id, "is_active": True},
        sort=[("updated_at", -1)],
    )

    return {
        "total_sessions": session_count,
        "total_questions": message_count,
        "last_active": last_session["updated_at"] if last_session else None,
    }