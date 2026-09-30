"""Streamlit chat UI for the ABC Agentic AI assistant."""
import os
import uuid
from pathlib import Path

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API = f"{BACKEND_URL}/api/v1"
ROBOT_AVATAR = str(Path(__file__).with_name("robot.svg"))

st.set_page_config(page_title="ABC Agentic AI Assistant", page_icon="🤖", layout="wide")


def new_thread() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


if "thread_id" not in st.session_state:
    new_thread()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_models() -> dict:
    try:
        r = requests.get(f"{API}/models", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return {"default": "gemini-3.5-flash", "models": ["gemini-3.5-flash", "gemini-3.5-flash-lite"]}


def fetch_health() -> dict | None:
    try:
        r = requests.get(f"{API}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    info = fetch_models()
    models = info["models"]
    model = st.selectbox(
        "Model",
        models,
        index=models.index(info["default"]) if info["default"] in models else 0,
    )
    if st.button("🔄 New conversation", use_container_width=True):
        new_thread()
        st.rerun()
    st.caption(f"Thread ID: `{st.session_state.thread_id[:8]}…`")

    st.divider()
    st.subheader("System status")
    health = fetch_health()
    if health is None:
        st.error("Backend unreachable")
    else:
        st.write(("✅" if health["database"] else "❌") + " PostgreSQL")
        st.write(("✅" if health["vector_store"] else "❌") + " ChromaDB")
        st.write(("✅" if health["api_key_configured"] else "❌") + " Gemini API key")
    if st.button("Refresh status", use_container_width=True):
        st.rerun()

    st.divider()
    st.subheader("Try asking")
    st.markdown(
        "- *How many days of annual leave do I get?*\n"
        "- *Can I work remotely from another country?*\n"
        "- *What's the status of order ORD-101?*\n"
        "- *Cancel order ORD-104*\n"
        "- *Cancel order ORD-102* (shipped → refused)"
    )

# ------------------------------------------------------------------- Chat
st.title("🤖 ABC Agentic AI Assistant")
st.markdown(
        """
        <div style="margin-top: -0.75rem; margin-bottom: 1.5rem; line-height: 1.35;">
            <div style="font-size: 0.875rem; color: var(--text-color); opacity: 0.65;">
                ABC Company policy Q&amp;A (RAG) + live order tools, powered by LangGraph.
            </div>
            <div style="margin-top: 0.15rem; color: var(--text-color); opacity: 0.65; font-size: 0.875rem;">
                Developed by:
                <a href="https://www.linkedin.com/in/abchatterjee7" target="_blank" rel="noopener noreferrer">
                    Aaditya B Chatterjee
                </a>
            </div>
        </div>
        """,
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    avatar = ROBOT_AVATAR if msg["role"] == "assistant" else "😊"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        meta = msg.get("meta")
        if meta:
            if meta.get("tool_used", "none") != "none":
                st.caption(f"🛠️ Tool: `{meta['tool_used']}`")
            if meta.get("sources"):
                st.caption("📚 Sources: " + "; ".join(meta["sources"]))

if prompt := st.chat_input("Ask about company policies or your orders…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="😊"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ROBOT_AVATAR):
        meta = None
        with st.spinner("Thinking…"):
            try:
                r = requests.post(
                    f"{API}/chat",
                    json={
                        "message": prompt,
                        "thread_id": st.session_state.thread_id,
                        "model": model,
                    },
                    timeout=120,
                )
                if r.ok:
                    data = r.json()
                    answer = data["response"]
                    meta = {"tool_used": data["tool_used"], "sources": data["sources"]}
                else:
                    detail = r.json().get("detail", r.text) if r.content else r.reason
                    answer = f"⚠️ Backend error ({r.status_code}): {detail}"
            except requests.RequestException as exc:
                answer = f"⚠️ Could not reach the backend: {exc}"
        st.markdown(answer)
        if meta:
            if meta["tool_used"] != "none":
                st.caption(f"🛠️ Tool: `{meta['tool_used']}`")
            if meta["sources"]:
                st.caption("📚 Sources: " + "; ".join(meta["sources"]))

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
