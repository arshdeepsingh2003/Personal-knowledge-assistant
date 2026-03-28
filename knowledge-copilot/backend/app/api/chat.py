import json
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import StreamingResponse

from app.services.retriever import retrieve, format_context_for_llm
from app.services.llm import generate_answer, stream_answer
from app.services.chat_session import (
    create_session, get_session, get_history,
    add_message, list_sessions, delete_session,
)
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/chat", tags=["chat"])

# This file implements a RAG (Retrieval-Augmented Generation) flow:
# User Query → Retrieve Docs → Build Context → LLM → Answer

# Session management(Creates a new chat session)

@router.post("/session")
def new_session():
    """Create a new chat session."""
    session_id = create_session()
    return {"session_id": session_id}


@router.get("/sessions")
def all_sessions():
    return {"sessions": list_sessions()}


@router.get("/session/{session_id}")
def get_session_detail(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/session/{session_id}")
def remove_session(session_id: str):
    ok = delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": f"Session {session_id} deleted"}


# Core chat endpoint (blocking) 
@router.post("/message")
def chat_message(
    session_id: str = Form(...),
    query:      str = Form(...),
    k:          int   = Form(5),
    score_threshold: float = Form(0.30),
):
    """
    Full RAG pipeline in one endpoint:
      1. Validate session
      2. Retrieve relevant chunks
      3. Build context
      4. Generate answer with history
      5. Save to session
      6. Return answer + sources
    """
    # 1. Validate session
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Create one first.")

    # 2. Check index is populated
    if get_vector_store().stats()["total_docs"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload and index a document first."
        )

    # 3. Retrieve
    result  = retrieve(query, k=k, score_threshold=score_threshold)
    context = format_context_for_llm(result)

    # 4. Get history and generate
    history = get_history(session_id)
    answer  = generate_answer(query, context, history)

    # 5. Persist both turns to session
    add_message(session_id, "user",      query)
    add_message(session_id, "assistant", answer)

    # 6. Return
    return {
        "session_id": session_id,
        "query":      query,
        "answer":     answer,
        "sources": [
            {
                "file_name":   s.file_name,
                "page":        s.page,
                "score":       s.score,
                "preview":     s.preview,
            }
            for s in result.sources
        ],
        "context_used": result.total_found > 0,
    }


# Streaming chat endpoint

@router.post("/stream")
def chat_stream(
    session_id: str = Form(...),
    query:      str = Form(...),
    k:          int   = Form(5),
    score_threshold: float = Form(0.30),
):
    """
    Same as /chat/message but streams the answer token-by-token via SSE.
    The frontend receives a text/event-stream response and renders
    each chunk as it arrives — like ChatGPT's typing effect.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if get_vector_store().stats()["total_docs"] == 0:
        raise HTTPException(status_code=400, detail="No documents indexed.")

    result  = retrieve(query, k=k, score_threshold=score_threshold)
    context = format_context_for_llm(result)
    history = get_history(session_id)

    # Save user message immediately
    add_message(session_id, "user", query)

    def event_generator():
        full_answer = []

        # Stream sources first as a JSON event so the UI can show them early
        sources_payload = json.dumps({
            "type": "sources",
            "sources": [
                {
                    "file_name": s.file_name,
                    "page":      s.page,
                    "score":     s.score,
                }
                for s in result.sources
            ]
        })
        yield f"data: {sources_payload}\n\n"

        # Stream answer tokens
        for token in stream_answer(query, context, history):
            full_answer.append(token)
            payload = json.dumps({"type": "token", "content": token})
            yield f"data: {payload}\n\n"

        # Save complete assistant message to session
        add_message(session_id, "assistant", "".join(full_answer))

        # Send a done event so the frontend knows to stop listening
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )