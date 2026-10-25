"""
Streamlit chat UI for the weather + search agent.

Start the API first, then this app:
    uvicorn api:app --reload --port 8000
    streamlit run streamlit_app.py

Set API_URL if the backend isn't on localhost:8000.
"""
from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120  # agent turns with several tool calls can be slow

st.set_page_config(page_title="Weather & Search Agent", page_icon="⛅")


def api_healthy() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=5).ok
    except requests.RequestException:
        return False

def new_conversation() -> None:
    """Start a fresh thread, discarding server-side state for the old one."""
    old_thread = st.session_state.get("thread_id")
    if old_thread:
        try:
            requests.delete(f"{API_URL}/history/{old_thread}", timeout=10)
        except requests.RequestException:
            pass  # server may already be gone; nothing to clean up locally
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    # Rehydrate from the API so a browser refresh doesn't lose the thread.
    try:
        response = requests.get(
            f"{API_URL}/history/{st.session_state.thread_id}", timeout=10
        )
        st.session_state.messages = response.json() if response.ok else []
    except requests.RequestException:
        st.session_state.messages = []

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Session")
    if api_healthy():
        st.success("API connected")
    else:
        st.error(f"No API at {API_URL}")
        st.caption("Start it with: `uvicorn api:app --reload --port 8000`")

    st.caption(f"Thread: `{st.session_state.thread_id[:8]}…`")
    st.button("New conversation", on_click=new_conversation, use_container_width=True)

    st.divider()
    st.markdown(
        "**This agent can:**\n"
        "- Look up current weather and forecasts (WeatherAPI)\n"
        "- Search the web for anything else (Tavily)\n"
        "- Remember earlier turns in this conversation"
    )

    
# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
st.title("⛅ Weather & Search Agent")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the weather, or anything else…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "message": prompt,
                        "thread_id": st.session_state.thread_id,
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                reply = response.json()["reply"]
            except requests.Timeout:
                reply = "The agent took too long to respond. Try again."
            except requests.RequestException as exc:
                reply = f"Could not reach the agent: {exc}"

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})