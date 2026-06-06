import streamlit as st
from typing import List


# Store uploaded PDF bytes in session for viewer access
def store_pdf(filename: str, file_bytes: bytes):
    """Store PDF bytes in session state for the viewer."""
    if "pdf_store" not in st.session_state:
        st.session_state["pdf_store"] = {}
    st.session_state["pdf_store"][filename] = file_bytes


def render_pdf_viewer():
    """
    Render the right panel: PDF viewer with bidirectional page navigation.
    
    Features:
    - Persistent page state (stays on page until user changes it)
    - Click citation button → jump to that page
    - Manual page input → navigate to any page (up or down)
    """
    st.markdown("### 📄 Viewer Dokumen")
    
    pdf_store = st.session_state.get("pdf_store", {})
    
    if not pdf_store:
        st.info("Upload PDF dari sidebar untuk mulai melihat dokumen.")
        return
    
    # ─── Initialize persistent viewing state ───────────────────
    if "viewing_pdf" not in st.session_state:
        st.session_state["viewing_pdf"] = list(pdf_store.keys())[0] if pdf_store else None
    if "viewing_page" not in st.session_state:
        st.session_state["viewing_page"] = 1
    
    # ─── Handle citation click ─────────────────────────────────
    # If user clicked a citation button, jump to that page
    if st.session_state.get("target_pdf"):
        st.session_state["viewing_pdf"] = st.session_state["target_pdf"]
        st.session_state["target_pdf"] = None  # Clear trigger
    if st.session_state.get("target_page"):
        st.session_state["viewing_page"] = st.session_state["target_page"]
        st.session_state["target_page"] = None  # Clear trigger
    
    # ─── PDF selector ──────────────────────────────────────────
    pdf_names = list(pdf_store.keys())
    if st.session_state["viewing_pdf"] not in pdf_names:
        st.session_state["viewing_pdf"] = pdf_names[0]
    
    selected_pdf = st.selectbox(
        "Pilih dokumen:",
        pdf_names,
        index=pdf_names.index(st.session_state["viewing_pdf"]),
        key="pdf_selector"
    )
    
    if not selected_pdf:
        return
    
    # Update viewing_pdf if user changed via dropdown
    st.session_state["viewing_pdf"] = selected_pdf
    
    # ─── Page navigation controls ──────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        if st.button("⬅️ Sebelumnya", use_container_width=True):
            if st.session_state["viewing_page"] > 1:
                st.session_state["viewing_page"] -= 1
                st.rerun()
    with col2:
        st.markdown(f"<div style='text-align: center; font-size: 1.2rem; padding-top: 4px;'>"
                    f"<b>Hal. {st.session_state['viewing_page']}</b></div>",
                    unsafe_allow_html=True)
    with col3:
        if st.button("Selanjutnya ➡️", use_container_width=True):
            st.session_state["viewing_page"] += 1
            st.rerun()
    
    # ─── Manual page input ─────────────────────────────────────
    go_col1, go_col2 = st.columns([3, 1])
    with go_col1:
        manual_page = st.number_input(
            "Lompat ke halaman:",
            min_value=1,
            max_value=9999,
            value=st.session_state["viewing_page"],
            step=1,
            key="manual_page_input",
            label_visibility="collapsed",
            placeholder="Nomor halaman..."
        )
    with go_col2:
        if st.button("Ke Halaman", use_container_width=True):
            if manual_page != st.session_state["viewing_page"]:
                st.session_state["viewing_page"] = manual_page
                st.rerun()
    
    # ─── Render PDF ────────────────────────────────────────────
    file_bytes = pdf_store[selected_pdf]
    
    try:
        from streamlit_pdf_viewer import pdf_viewer
        
        current_page = st.session_state["viewing_page"]
        # Render some pages around current page for context
        pages_to_render = list(range(
            max(1, current_page - 1),
            current_page + 5
        ))
        
        pdf_viewer(
            input=file_bytes,
            width=700,
            height=700,
            scroll_to_page=current_page,
            pages_to_render=pages_to_render,
            key=f"pdf_{selected_pdf}_{current_page}"
        )
        
    except ImportError:
        st.warning("PDF viewer memerlukan library streamlit-pdf-viewer.")
        st.markdown(f"**Dokumen:** {selected_pdf}")
        st.info(f"📍 **Halaman {st.session_state['viewing_page']}**")
        st.info("Untuk melihat PDF, install: `pip install streamlit-pdf-viewer`")
    
    except Exception as e:
        st.error(f"Gagal menampilkan PDF: {str(e)[:200]}")