import streamlit as st
from typing import List, Any
import uuid
import json
import os

CHATS_FILE = "data/chats.json"

def save_chats():
    os.makedirs(os.path.dirname(CHATS_FILE), exist_ok=True)
    with open(CHATS_FILE, "w") as f:
        json.dump({
            "chats": st.session_state.chats,
            "current_chat_id": st.session_state.current_chat_id
        }, f)

def load_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, "r") as f:
                data = json.load(f)
                return data.get("chats", {}), data.get("current_chat_id")
        except Exception:
            return None, None
    return None, None

def init_session_state(settings):
    # Core settings
    if "top_k" not in st.session_state:
        st.session_state.top_k = settings.top_k
    if "temperature" not in st.session_state:
        st.session_state.temperature = settings.temperature
    if "model_name" not in st.session_state:
        st.session_state.model_name = settings.model_name
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False

    # Multi-chat management with persistence
    if "chats" not in st.session_state:
        saved_chats, saved_current_id = load_chats()
        if saved_chats:
            st.session_state.chats = saved_chats
            st.session_state.current_chat_id = saved_current_id
        else:
            chat_id = str(uuid.uuid4())
            st.session_state.chats = {
                chat_id: {
                    "name": "New Chat",
                    "messages": [],
                    "selected_papers": [],
                    "selected_ids": []
                }
            }
            st.session_state.current_chat_id = chat_id

def display_chat():
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    for message in current_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def show_retrieved(docs: List[Any]):
    if not docs:
        return
    with st.expander("🔍 Retrieved Chunks", expanded=False):
        for i, (doc, score) in enumerate(docs):
            st.markdown(f"**Chunk {i+1}** (Score: {score:.4f})")
            st.markdown(f"*Source: {doc.metadata.get('title')}*")
            st.text(doc.page_content)
            st.divider()
