import streamlit as st
from typing import List, Dict


def stream_markdown(text: str, *, chunk_size: int = 8, delay_s: float = 0.01):
    """Pseudo-stream markdown text to improve perceived responsiveness.

    This does NOT require provider-side token streaming.
    It progressively renders the final text in small chunks.

    Notes:
    - Uses st.empty() placeholder.
    - Avoids too frequent rerenders by chunking.
    """
    import time

    if text is None:
        text = ""
    text = str(text)

    placeholder = st.empty()
    acc = ""
    for i in range(0, len(text), chunk_size):
        acc += text[i : i + chunk_size]
        placeholder.markdown(acc)
        time.sleep(delay_s)
    return placeholder


def render_chat_panel():
    """
    Render the left panel: Chat interface with message history and question input.
    
    Each assistant message shows citations as clickable buttons
    that trigger PDF page jumping in the right panel.
    """
    
    # Chat message container
    chat_container = st.container(border=True, height=500)
    
    with chat_container:
        # Display chat history
        for msg_idx, msg in enumerate(st.session_state.get("messages", [])):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # If assistant message with citations, show citation buttons
                if msg["role"] == "assistant" and msg.get("citations"):
                    _render_citation_buttons(msg["citations"], msg_idx=msg_idx)
    
    # Chat input
    user_query = st.chat_input("Tanyakan tentang dokumen Anda...")
    
    return user_query


def _render_citation_buttons(citations: List[Dict], msg_idx: int = 0):
    """
    Render clickable citation buttons for each source.
    
    Args:
        citations: List of dicts with keys: page, filename, snippet
        msg_idx: Index of the message in chat history (for unique keys)
    """
    if not citations:
        return
    
    st.markdown("---")
    st.markdown("**📚 Sumber:**")
    
    cols = st.columns(min(len(citations), 3))
    
    for i, citation in enumerate(citations):
        col_idx = i % 3
        with cols[col_idx]:
            page = citation.get("page", "?")
            filename = citation.get("filename", "dokumen")
            
            button_label = f"📄 Hal.{page}"
            
            if st.button(
                button_label,
                key=f"cite_{msg_idx}_{i}_{page}",
                use_container_width=True,
                help=f"Buka halaman {page} dari {filename}"
            ):
                # Update session state for PDF viewer page jump
                st.session_state["target_pdf"] = filename
                st.session_state["target_page"] = page
                st.rerun()


def add_user_message(message: str):
    """Add a user message to the chat history."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    st.session_state["messages"].append({
        "role": "user",
        "content": message
    })


def add_assistant_message(answer: str, citations: List[Dict] = None):
    """Add an assistant message with optional citations to the chat history."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "citations": citations or []
    })


def stream_assistant_message(answer: str, citations: List[Dict] = None):
    """Stream-render assistant message, then persist it to chat history."""
    with st.chat_message("assistant"):
        stream_markdown(answer)
        if citations:
            _render_citation_buttons(citations, msg_idx=len(st.session_state.get("messages", [])))

    # Persist after streaming so it won't stream again on rerun
    add_assistant_message(answer, citations)