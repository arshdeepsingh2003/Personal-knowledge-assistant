import json
from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import StreamingResponse

from app.middleware.auth_middleware import get_current_user
from app.services.chat_history import (
    create_session, get_session, list_sessions,
    delete_session, save_user_message, save_assistant_message,
    get_messages, get_history_for_llm, get_user_stats,
)
from app.services.retriever import retrieve, format_context_for_llm
from app.services.llm import generate_answer, stream_answer
from app.services.vector_store import get_vector_store
from app.core.config import settings

router = APIRouter(prefix="/chat", tags=["chat"])


# ── Session management ────────────────────────────────────────────────────────

@router.post("/session")
async def new_session(
    current_user: dict = Depends(get_current_user)
):
    session_id = await create_session(user_id=current_user["id"])
    return {"session_id": session_id}


@router.get("/sessions")
async def all_sessions(
    current_user: dict = Depends(get_current_user)
):
    sessions = await list_sessions(user_id=current_user["id"])
    return {"sessions": sessions}


@router.get("/session/{session_id}")
async def session_detail(
    session_id:   str,
    current_user: dict = Depends(get_current_user),
):
    session = await get_session(session_id, user_id=current_user["id"])
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Include full message history
    messages = await get_messages(session_id, user_id=current_user["id"])
    return {**session, "messages": messages}


@router.delete("/session/{session_id}")
async def remove_session(
    session_id:   str,
    current_user: dict = Depends(get_current_user),
):
    ok = await delete_session(session_id, user_id=current_user["id"])
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted"}


@router.get("/stats")
async def user_stats(
    current_user: dict = Depends(get_current_user)
):
    return await get_user_stats(user_id=current_user["id"])


# ── Core chat endpoint ────────────────────────────────────────────────────────

@router.post("/message")
async def chat_message(
    session_id:      str   = Form(...),
    query:           str   = Form(...),
    k:               int   = Form(5),
    score_threshold: float = Form(0.30),
    current_user:    dict  = Depends(get_current_user),
):
    """
    Full RAG pipeline with persistent history:
      1. Verify session ownership
      2. Retrieve relevant chunks
      3. Load conversation history from MongoDB
      4. Generate answer
      5. Save both messages to MongoDB with full context
      6. Return answer + sources
    """
    # 1. Verify session belongs to this user
    session = await get_session(session_id, user_id=current_user["id"])
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Create one first via POST /chat/session",
        )

    # 2. Check index
    if get_vector_store().stats()["total_docs"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet.",
        )

    # 3. Retrieve context
    result  = retrieve(query, k=k, score_threshold=score_threshold)
    context = format_context_for_llm(result)

    # 4. Get conversation history from MongoDB
    history = await get_history_for_llm(session_id, user_id=current_user["id"])

    # 5. Generate answer
    answer = generate_answer(query, context, history)

    # 6. Save to MongoDB
    await save_user_message(
        session_id = session_id,
        user_id    = current_user["id"],
        content    = query,
    )
    await save_assistant_message(
        session_id     = session_id,
        user_id        = current_user["id"],
        content        = answer,
        sources        = [
            {
                "file_name": s.file_name,
                "page":      s.page,
                "score":     s.score,
                "preview":   s.preview,
            }
            for s in result.sources
        ],
        context_chunks = result.chunks,   # full RAG context stored for auditing
        model          = settings.llm_model,
    )

    return {
        "session_id":   session_id,
        "query":        query,
        "answer":       answer,
        "sources": [
            {
                "file_name": s.file_name,
                "page":      s.page,
                "score":     s.score,
                "preview":   s.preview,
            }
            for s in result.sources
        ],
        "context_used": result.total_found > 0,
    }


# ── Streaming endpoint ────────────────────────────────────────────────────────

@router.post("/stream")
async def chat_stream(
    session_id:      str   = Form(...),
    query:           str   = Form(...),
    k:               int   = Form(5),
    score_threshold: float = Form(0.30),
    current_user:    dict  = Depends(get_current_user),
):
    session = await get_session(session_id, user_id=current_user["id"])
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if get_vector_store().stats()["total_docs"] == 0:
        raise HTTPException(status_code=400, detail="No documents indexed.")

    result  = retrieve(query, k=k, score_threshold=score_threshold)
    context = format_context_for_llm(result)
    history = await get_history_for_llm(session_id, user_id=current_user["id"])

    # Save user message before streaming starts
    await save_user_message(
        session_id = session_id,
        user_id    = current_user["id"],
        content    = query,
    )

    sources_payload = [
        {"file_name": s.file_name, "page": s.page, "score": s.score}
        for s in result.sources
    ]

    async def event_generator():
        full_answer = []

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload})}\n\n"

        for token in stream_answer(query, context, history):
            full_answer.append(token)
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # Save complete assistant message after stream ends
        complete_answer = "".join(full_answer)
        await save_assistant_message(
            session_id     = session_id,
            user_id        = current_user["id"],
            content        = complete_answer,
            sources        = sources_payload,
            context_chunks = result.chunks,
            model          = settings.llm_model,
        )

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )