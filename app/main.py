import os
# Fix for Protobuf Descriptor Error - Must be set before any other imports
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
from pathlib import Path
import uuid
import streamlit as st
import random

# Setup project root
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from config.config import settings
from rag.ingest import ingest
from rag.retriever import ArxivRetriever
from rag.chain import create_conversational_chain
from rag.analysis import sample_papers_by_category
from app.ui.components import init_session_state, display_chat, show_retrieved, save_chats

def main():
    st.set_page_config(page_title="arXiv StudyPal", layout="wide", page_icon="📚")
    init_session_state(settings)

    # --- Sidebar ---
    with st.sidebar:
        st.title("📚 arXiv StudyPal")
        
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.chats[new_id] = {
                "name": "New Chat",
                "messages": [],
                "selected_papers": [],
                "selected_ids": []
            }
            st.session_state.current_chat_id = new_id
            save_chats()
            st.rerun()

        st.markdown("---")
        st.subheader("💬 Your Chats")
        
        # Chat selection and deletion
        for chat_id in list(st.session_state.chats.keys()):
            chat_data = st.session_state.chats[chat_id]
            col1, col2 = st.columns([0.8, 0.2])
            
            is_current = (chat_id == st.session_state.current_chat_id)
            label = f"{'⭐ ' if is_current else ''}{chat_data['name']}"
            
            if col1.button(label, key=f"chat_btn_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                save_chats()
                st.rerun()
            
            if col2.button("🗑️", key=f"del_{chat_id}", help="Delete chat"):
                # Always allow deletion; if it's the last one, we'll create a new empty one
                del st.session_state.chats[chat_id]
                
                # If we deleted all chats, create a new one immediately
                if not st.session_state.chats:
                    new_id = str(uuid.uuid4())
                    st.session_state.chats[new_id] = {
                        "name": "New Chat",
                        "messages": [],
                        "selected_papers": [],
                        "selected_ids": []
                    }
                    st.session_state.current_chat_id = new_id
                # If we deleted the current chat but others remain, switch to the first available
                elif st.session_state.current_chat_id == chat_id:
                    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                
                save_chats()
                st.rerun()

        st.markdown("---")
        st.subheader("🔍 Browse Papers")
        categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"]
        selected_cat = st.selectbox("Category", categories)
        
        papers = sample_papers_by_category(selected_cat, limit=50)
        paper_options = {p['title']: p['arxiv_id'] for p in papers}
        
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        
        # Ensure previously selected papers are ALWAYS in the options list
        # even if they aren't in the currently sampled 50 for this category
        for title, paper_id in zip(current_chat["selected_papers"], current_chat["selected_ids"]):
            if title not in paper_options:
                paper_options[title] = paper_id
        
        selected_paper_titles = st.multiselect(
            "Select Papers for this Chat",
            options=list(paper_options.keys()),
            default=current_chat["selected_papers"],
            key=f"papers_{st.session_state.current_chat_id}"
        )
        
        # Update current chat state ONLY if changed
        if selected_paper_titles != current_chat["selected_papers"]:
            current_chat["selected_papers"] = selected_paper_titles
            current_chat["selected_ids"] = [paper_options[t] for t in selected_paper_titles]
            
            # Update chat name if it's still "New Chat" and papers are selected
            if current_chat["name"] == "New Chat" and selected_paper_titles:
                current_chat["name"] = selected_paper_titles[0][:20] + "..." if len(selected_paper_titles[0]) > 20 else selected_paper_titles[0]
            
            save_chats()
            st.rerun()

        with st.expander("🛠️ Advanced", expanded=False):
            st.session_state.show_debug = st.checkbox("Show Retrieved Chunks", st.session_state.show_debug)

    # --- Main Chat UI ---
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    st.title(f"💬 {current_chat['name']}")

    # Learning Mode Header
    if current_chat["selected_papers"]:
        st.info(f"**Learning from:**\n" + "\n".join([f"- {p}" for p in current_chat["selected_papers"]]))
    else:
        st.warning("⚠️ Please select at least one paper in the sidebar to start learning.")

    # Display chat history
    display_chat()

    # Chat Input Handling
    query = st.chat_input("Ask a question about the selected papers...")
    
    # Check for pending query from buttons
    if hasattr(st.session_state, 'pending_query'):
        query = st.session_state.pending_query
        del st.session_state.pending_query

    # Process Query
    if query:
        if not current_chat["selected_ids"]:
            st.error("Please select at least one paper first!")
        else:
            # Add user message
            current_chat["messages"].append({"role": "user", "content": query})
            save_chats() 
            with st.chat_message("user"):
                st.markdown(query)

            # Generate response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    try:
                        chain = create_conversational_chain(
                            llm_model=st.session_state.model_name,
                            temperature=st.session_state.temperature,
                            arxiv_ids=current_chat["selected_ids"]
                        )
                        
                        # Use invoke for modern LangChain
                        result = chain.invoke({"question": query, "chat_history": []})
                        answer = result['answer']
                        retrieved_docs = [(doc, 0.0) for doc in result.get('source_documents', [])]
                        
                        # Simpler character-by-character streaming for better UX
                        full_response = ""
                        for char in answer:
                            full_response += char
                            response_placeholder.markdown(full_response + "▌")
                            import time
                            # Only sleep for the first few hundred chars for snappy feel
                            if len(full_response) < 500:
                                time.sleep(0.002)
                        response_placeholder.markdown(full_response)
                        
                        current_chat["messages"].append({"role": "assistant", "content": full_response})
                        save_chats() 
                        
                        if st.session_state.show_debug:
                            show_retrieved(retrieved_docs)
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # Suggested Quick Follow-ups (Rendered after current chat output)
    if current_chat["selected_papers"]:
        st.markdown("---")
        st.markdown("### 💡 Quick Follow-ups")
        
        q_cols = st.columns(2)
        main_paper = current_chat['selected_papers'][0]
        num_messages = len(current_chat["messages"])
        
        # Use a seed based on chat_id and message count to keep suggestions 
        # stable for the current turn but different across turns
        random.seed(f"{st.session_state.current_chat_id}_{num_messages}")
        
        if num_messages == 0:
            pool = [
                f"Summarize the core method in '{main_paper[:30]}...'",
                "What are the main experimental results?",
                "What problem does this paper solve?",
                "What are the key contributions?",
                "What is the significance of the title?",
                "Who are the target audiences for this research?"
            ]
            if len(current_chat["selected_papers"]) > 1:
                pool.append("What is the common theme between these papers?")
                pool.append("Compare the key contributions of the selected papers.")
                pool.append("How do these papers build upon each other?")
        elif num_messages < 4:
            pool = [
                "Explain the limitations discussed.",
                "How does this work compare to state-of-the-art?",
                "What are the potential future research directions?",
                "Are there any specific datasets or benchmarks mentioned?",
                "What were the main evaluation metrics used?",
                "Explain the ablation studies if any.",
                "What are the core assumptions made by the authors?",
                "Describe the hardware or compute resources used."
            ]
        else:
            pool = [
                "Can you explain the mathematical foundation?",
                "What are the practical implications of this research?",
                "Summarize the final conclusions.",
                "Give me a 3-sentence elevator pitch for this work.",
                "How can this be applied in industry?",
                "What are the ethical considerations mentioned?",
                "If I were to replicate this, where should I start?",
                "What is the most surprising finding in this paper?"
            ]

        # Randomly select 4 questions from the pool
        selected_queries = random.sample(pool, min(4, len(pool)))

        for i, ex in enumerate(selected_queries):
            if q_cols[i % 2].button(ex, use_container_width=True, key=f"suggested_{i}_{st.session_state.current_chat_id}_{num_messages}"):
                st.session_state.pending_query = ex
                st.rerun()

if __name__ == "__main__":
    main()
