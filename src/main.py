import sys
import os

# Ensure the project root is in sys.path for Streamlit execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from typing import Dict, Optional

from src.config import APP_TITLE, APP_ICON, get_model_display_names, get_default_model
from src.ingestion.pdf_loader import extract_pdf_text
from src.ingestion.chunker import chunk_documents
from src.vector_store.faiss_db import FAISSDocumentStore
from src.graph.workflow import set_vector_store, run_workflow
from src.ui.chat_panel import render_chat_panel, add_user_message, stream_assistant_message
from src.ui.pdf_viewer import render_pdf_viewer, store_pdf


def _get_current_pdf_pages_for_vision() -> Dict[int, "Image.Image"]:
    """Build a {page_num: PIL.Image} dict for the currently selected PDF in the viewer.

    This is used by the image agent (OpenRouter vision) when user asks about a specific page image.
    We render pages lazily and cache them in session_state.
    """
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        return {}

    pdf_store = st.session_state.get("pdf_store", {})
    if not pdf_store:
        return {}

    # pick current viewing pdf if exists, else first uploaded
    viewing_pdf = st.session_state.get("viewing_pdf")
    if not viewing_pdf or viewing_pdf not in pdf_store:
        viewing_pdf = list(pdf_store.keys())[0]

    file_bytes = pdf_store.get(viewing_pdf)
    if not file_bytes:
        return {}

    # cache: pdf_page_images[filename] = {page_num: PIL.Image}
    if "pdf_page_images" not in st.session_state:
        st.session_state["pdf_page_images"] = {}
    if viewing_pdf not in st.session_state["pdf_page_images"]:
        st.session_state["pdf_page_images"][viewing_pdf] = {}

    # Render all pages (safe for small PDFs); can be optimized later.
    # We only render if cache is empty.
    if not st.session_state["pdf_page_images"][viewing_pdf]:
        try:
            import io
            import pypdfium2 as pdfium
            from src.config import RENDER_DPI

            pdf = pdfium.PdfDocument(io.BytesIO(file_bytes))
            for i in range(len(pdf)):
                page = pdf[i]
                bitmap = page.render(scale=RENDER_DPI / 72)
                img = bitmap.to_pil()
                st.session_state["pdf_page_images"][viewing_pdf][i + 1] = img
                page.close()
            pdf.close()
        except Exception:
            return {}

    return st.session_state["pdf_page_images"][viewing_pdf]


# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(f"{APP_ICON} {APP_TITLE}")
st.markdown("Upload dokumen PDF laporan keuangan dan ajukan pertanyaan. AI akan menjawab berdasarkan teks dalam dokumen.")


# ─── Initialize Session State ────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables if they don't exist."""
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = FAISSDocumentStore()
        set_vector_store(st.session_state["vector_store"])
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    if "pdf_store" not in st.session_state:
        st.session_state["pdf_store"] = {}
    
    if "target_pdf" not in st.session_state:
        st.session_state["target_pdf"] = None
    
    if "target_page" not in st.session_state:
        st.session_state["target_page"] = None
    
    if "is_processing" not in st.session_state:
        st.session_state["is_processing"] = False
    
    if "uploaded_files_list" not in st.session_state:
        st.session_state["uploaded_files_list"] = []
    
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = get_default_model()


# ─── Sidebar: PDF Upload ─────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar with PDF upload and document management."""
    with st.sidebar:
        st.markdown("## 📤 Upload Dokumen")
        st.markdown("**Format:** PDF Laporan Keuangan")
        
        uploaded_files = st.file_uploader(
            "Pilih file PDF",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        enable_vision = st.toggle(
            "🔍 Gunakan Vision untuk Tabel", 
            value=True, 
            help="Sangat direkomendasikan untuk laporan keuangan agar tabel terekstrak sempurna."
        )
        
        if uploaded_files:
            with st.status("🔄 Memproses dokumen...", expanded=True) as status:
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    
                    # Skip if already processed
                    if file_name in st.session_state["uploaded_files_list"]:
                        st.write(f"⏭️ {file_name} — sudah diproses sebelumnya")
                        continue
                    
                    try:
                        # Read file bytes
                        file_bytes = uploaded_file.getvalue()
                        
                        # Store PDF for viewer
                        store_pdf(file_name, file_bytes)
                        
                        # Extract text from PDF
                        st.write(f"📖 Mengekstrak teks dari: {file_name}")
                        pages = extract_pdf_text(
                            file_bytes, 
                            file_name, 
                            enable_table_vision=enable_vision
                        )
                        
                        # Chunk the documents
                        st.write(f"✂️ Memotong teks menjadi chunks...")
                        chunks = chunk_documents(pages)
                        
                        # Add to vector store
                        st.write(f"🧠 Menambahkan {len(chunks)} chunks ke vector store...")
                        count = st.session_state["vector_store"].add_documents(chunks)
                        
                        # Track processed files
                        st.session_state["uploaded_files_list"].append(file_name)
                        
                        st.write(f"✅ {file_name} — {count} chunks diproses")
                        
                    except Exception as e:
                        st.error(f"❌ Gagal memproses {file_name}: {str(e)[:200]}")
                
                status.update(label="✅ Semua dokumen selesai diproses", state="complete")
        
        # Show document stats
        total_chunks = st.session_state["vector_store"].total_documents
        total_files = len(st.session_state["uploaded_files_list"])
        
        st.sidebar.markdown("---")
        
        # ─── Model Selector ─────────────────────────────────────
        st.sidebar.markdown("### 🧠 Model AI")
        
        model_names = get_model_display_names()
        selected_idx = model_names.index(st.session_state["selected_model"]) if st.session_state["selected_model"] in model_names else 0
        
        selected_model = st.sidebar.selectbox(
            "Pilih Model Gemini:",
            options=model_names,
            index=selected_idx,
            key="model_selector",
            help="Ganti model AI untuk jawaban. Perubahan berlaku untuk pertanyaan berikutnya."
        )
        
        # Update session state if model changed
        if selected_model != st.session_state["selected_model"]:
            st.session_state["selected_model"] = selected_model
            st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### 📊 Statistik")
        st.sidebar.markdown(f"- **Dokumen terproses:** {total_files}")
        st.sidebar.markdown(f"- **Total chunks:** {total_chunks}")
        
        # Reset button
        if total_chunks > 0:
            if st.sidebar.button("🗑️ Reset Semua Dokumen", use_container_width=True):
                st.session_state["vector_store"].clear()
                st.session_state["uploaded_files_list"] = []
                st.session_state["messages"] = []
                st.session_state["pdf_store"] = {}
                st.rerun()


# ─── Main Application Layout ─────────────────────────────────────────
def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    
    # Two-column layout: Chat (left) | PDF Viewer (right)
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("## 💬 Chat")
        
        # Check if documents are loaded
        if st.session_state["vector_store"].total_documents == 0:
            st.info("Silakan upload dokumen PDF terlebih dahulu di sidebar sebelah kiri.")
        
        # Render chat panel and get user input
        user_query = render_chat_panel()
        
        # Process query if submitted
        if user_query and not st.session_state["is_processing"]:
            st.session_state["is_processing"] = True
            
            # Add user message to chat
            add_user_message(user_query)
            
            try:
                # Run the LangGraph workflow with selected model
                with st.spinner(f"🧠 Menganalisis dokumen dengan {st.session_state['selected_model']}..."):
                    result = run_workflow(
                        user_query, 
                        selected_model=st.session_state["selected_model"],
                        # NOTE: for image questions we need per-page rendered images.
                        # For now we render on-demand from the currently viewed PDF (fallback to first uploaded file).
                        pdf_pages=_get_current_pdf_pages_for_vision()
                    )
                
                # Stream assistant response to chat (then persist)
                stream_assistant_message(
                    result.get("final_answer") or "(Jawaban kosong — terjadi kesalahan pemetaan output. Silakan coba lagi.)",
                    result.get("final_citations") or result.get("draft_citations", [])
                )
                
                # Log retry count for transparency
                retry_count = result.get("retry_count", 0)
                review_result = result.get("review_result", {})
                if retry_count > 0:
                    st.caption(f"_Self-correction: {retry_count}x revisi | "
                               f"Status: {'✅ Lulus' if review_result.get('passed') else '❌ Dipaksa output'} _")
                
            except Exception as e:
                error_msg = f"Maaf, terjadi kesalahan sistem: {str(e)[:200]}"
                stream_assistant_message(error_msg)
            
            st.session_state["is_processing"] = False
            st.rerun()
    
    with col_right:
        render_pdf_viewer()


# ─── Entry Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    main()