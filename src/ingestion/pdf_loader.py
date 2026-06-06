import re
import io
from typing import List, Dict, Optional, Tuple
from PIL import Image

import fitz
import pymupdf4llm
import pypdfium2 as pdfium

from src.config import (
    GOOGLE_API_KEY,
    LLAMA_CLOUD_API_KEY,
    MIN_TEXT_LENGTH_FOR_TEXT_PAGE,
    VISION_DESCRIPTION_PROMPT,
    RENDER_DPI,
    ENABLE_VISION_INGESTION,
)
from src.ingestion.table_parser import extract_all_tables


def _extract_text_markdown(file_bytes: bytes) -> List[Dict]:
    """
    Extract text from PDF using PyMuPDF4LLM.
    
    Converts PDF directly to Markdown, natively preserving table structures
    as Markdown tables.
    
    Returns:
        List of {page_num, text, filename}
    """
    pages_data = []
    
    try:
        # Load doc from bytes
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        # Extract with page chunks
        md_pages = pymupdf4llm.to_markdown(doc, page_chunks=True)
        
        for i, md_page in enumerate(md_pages):
            # md_page metadata usually has 'page' (0-indexed) or we can use i+1
            page_idx = md_page.get("metadata", {}).get("page", i)
            pages_data.append({
                "page_num": page_idx + 1,
                "text": md_page.get("text", ""),
                "filename": ""
            })
            
    except Exception as e:
        raise RuntimeError(f"Markdown extraction failed: {str(e)}")
    
    return pages_data


def _render_page_to_image(file_bytes: bytes, page_num: int) -> Optional[Image.Image]:
    """Render PDF page as PIL Image."""
    try:
        pdf = pdfium.PdfDocument(io.BytesIO(file_bytes))
        
        if page_num >= len(pdf):
            pdf.close()
            return None
        
        page = pdf[page_num]
        bitmap = page.render(scale=RENDER_DPI / 72)
        img = bitmap.to_pil()
        
        page.close()
        pdf.close()
        
        return img
    except Exception as e:
        print(f"  ⚠️ Render halaman {page_num+1} gagal: {str(e)[:100]}")
        return None


def _describe_image_with_gemini(image: Image.Image) -> str:
    """Send image to Gemini Vision for description."""
    if not GOOGLE_API_KEY:
        return "[Gambar: Deskripsi tidak tersedia (GOOGLE_API_KEY tidak disetel)]"
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        
        response = model.generate_content([
            VISION_DESCRIPTION_PROMPT,
            image
        ])
        
        return response.text or "[Gambar: Tidak ada deskripsi yang dihasilkan]"
        
    except Exception as e:
        return f"[Gambar: Gagal mendeskripsikan - {str(e)[:100]}]"


def _extract_text_llamaparse(file_bytes: bytes, filename: str) -> List[Dict]:
    """
    Extract text using LlamaParse (cloud API) for perfect borderless table rendering.
    """
    import tempfile
    import os
    try:
        from llama_parse import LlamaParse
    except ImportError:
        raise RuntimeError("llama-parse library is not installed. Please install it to use LlamaParse.")

    pages_data = []
    
    # LlamaParse needs a file path, so we write bytes to a temp file
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(file_bytes)
            
        print("  ☁️ Mengirim dokumen ke LlamaParse (Cloud)...")
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            verbose=True
        )
        
        # load_data returns a list of Document objects
        documents = parser.load_data(temp_path)
        
        # We might just get 1 large document back, or 1 per page.
        # We will split by page if possible, but if not we'll treat it as one chunk.
        # Actually, let's just append each document returned.
        for i, doc in enumerate(documents):
            pages_data.append({
                "page_num": i + 1,
                "text": doc.text,
                "filename": filename
            })
            
    except Exception as e:
        raise RuntimeError(f"LlamaParse extraction failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return pages_data


def extract_pdf_text(file_bytes: bytes, filename: str, enable_table_vision: bool = True) -> List[Dict]:
    """
    Extract text from PDF using hybrid approach:
    - LlamaParse (if API Key provided) OR PyMuPDF4LLM untuk halaman teks + tabel
    - Gemini Vision untuk ekstraksi tabel kompleks (P0 Enhanced Table Parsing)
    - Gemini Vision untuk halaman gambar/diagram
    
    Tables are tagged with [TABLE_N] prefix and | separators.
    
    Args:
        file_bytes: Raw PDF file bytes.
        filename: Source PDF filename.
        enable_table_vision: If True (default), run Vision-based table parser
                             on pages detected as table-heavy.
    
    Returns:
        List of page dicts with 'page_num', 'text', 'filename'.
        Table chunks are APPENDED as additional pseudo-pages so the chunker
        can index them as [TABLE_N] blocks.
    """
    print(f"\n📄 Memproses: {filename}")
    
    if LLAMA_CLOUD_API_KEY:
        try:
            pages_data = _extract_text_llamaparse(file_bytes, filename)
        except Exception as e:
            print(f"  ⚠️ LlamaParse gagal ({e}), jatuh kembali ke PyMuPDF4LLM...")
            pages_data = _extract_text_markdown(file_bytes)
    else:
        pages_data = _extract_text_markdown(file_bytes)
    
    for page in pages_data:
        page["filename"] = filename
    
    total_pages = len(pages_data)
    image_pages_detected = 0
    table_pages_processed = 0
    
    # ── Step 1: Vision-based table extraction (P0) ──────────────
    if enable_table_vision and GOOGLE_API_KEY:
        try:
            table_chunks = extract_all_tables(file_bytes, pages_data, filename, force_all=False)
            if table_chunks:
                print(f"  📊 Total {len(table_chunks)} tabel diekstrak via Gemini Vision")
                table_pages_processed = len(table_chunks)
                
                # Append table chunks as pseudo-pages for the chunker.
                # The chunker already handles [TABLE_N] tags and assigns stable indices.
                for tc in table_chunks:
                    pages_data.append({
                        "page_num": tc["metadata"]["page_num"],
                        "text": tc["text"],
                        "filename": filename,
                        "_is_table_chunk": True,  # Marker for chunker
                        "_table_index": tc["metadata"]["table_index"],
                        "_table_metadata": tc["metadata"],
                    })
        except Exception as e:
            print(f"  ⚠️ Table extraction via Vision gagal: {e}")
    else:
        print(f"  ⏭️ Table vision: skipped (enable_table_vision={enable_table_vision}, API_KEY={'✅' if GOOGLE_API_KEY else '❌'})")
    
    # ── Step 2: Image page description (existing) ───────────────
    for i, page in enumerate(pages_data):
        # Skip table pseudo-pages (they don't need image description)
        if page.get("_is_table_chunk"):
            continue
        
        text_len = len(page["text"])
        
        if text_len < MIN_TEXT_LENGTH_FOR_TEXT_PAGE:
            image_pages_detected += 1
            print(f"  🖼️ Halaman {page['page_num']}: teks {text_len} karakter — "
                  f"{'mendeskripsikan dengan Gemini Vision...' if ENABLE_VISION_INGESTION else 'lewati deskripsi (vision ingestion OFF)'}")
            
            if ENABLE_VISION_INGESTION:
                img = _render_page_to_image(file_bytes, i)
                
                if img:
                    description = _describe_image_with_gemini(img)
                    page["text"] = f"[DESKRIPSI GAMBAR Halaman {page['page_num']}]: {description}"
                    print(f"     ✅ Deskripsi: {description[:80]}...")
                else:
                    page["text"] = f"[Gambar Halaman {page['page_num']}: Gagal dirender]"
            else:
                page["text"] = page["text"] or ""
        else:
            if not page.get("_is_table_chunk"):
                print(f"  📝 Halaman {page['page_num']}: teks {text_len} karakter — Markdown mode")
    
    # Cleanup internal markers before returning
    for page in pages_data:
        page.pop("_is_table_chunk", None)
        page.pop("_table_index", None)
        page.pop("_table_metadata", None)
    
    print(f"  📊 Ringkasan: {total_pages} halaman sumber, "
          f"{table_pages_processed} tabel via Vision, "
          f"{image_pages_detected} gambar terdeteksi")
    
    return pages_data
