"""
FastAPI backend for the weather + search agent.

Run:
    uvicorn api:app --reload --port 8000

Docs:
    http://localhost:8000/docs

Endpoints:
    POST   /chat                  send a message, get a reply
    GET    /history/{thread_id}   fetch the stored transcript
    DELETE /history/{thread_id}   wipe a conversation
    GET    /health                liveness probe
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import ask, clear_thread, get_history

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Weather & Search Agent",
    description="LangChain agent with WeatherAPI and Tavily tools.",
    version="1.0.0",
)

# Streamlit runs on a different port, so the browser treats it as a separate
# origin. Narrow this list before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message.")
    thread_id: str | None = Field(
        None,
        description="Conversation ID. Omit to start a new one; the response "
        "returns the generated ID to reuse on later turns.",
    )


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class Message(BaseModel):
    role: str
    content: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the agent within a conversation thread."""
    thread_id = request.thread_id or str(uuid.uuid4())
    try:
        reply = ask(request.message, thread_id=thread_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent invocation failed")
        raise HTTPException(
            status_code=502, detail=f"Agent failed: {exc}"
        ) from exc
    return ChatResponse(reply=reply, thread_id=thread_id)


@app.get("/history/{thread_id}", response_model=list[Message])
def history(thread_id: str) -> list[Message]:
    """Return the visible transcript for a conversation (empty if unknown)."""
    return [Message(**m) for m in get_history(thread_id)]


@app.delete("/history/{thread_id}", status_code=204)
def delete_history(thread_id: str) -> None:
    """Delete all stored state for a conversation."""
    clear_thread(thread_id)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}