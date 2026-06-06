import json
import base64
from typing import Optional, List
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import get_llm_vision_gemini
from src.graph.state import GraphState, Citation


def _extract_text_from_response(response) -> str:
    """Extract plain text from a Gemini/LangChain response.

    Gemini via LangChain can return:
    - AIMessage with .content as string
    - AIMessage with .content as list[dict] ("type":"text")
    """
    if hasattr(response, "text") and response.text:
        return response.text

    raw = response.content if hasattr(response, "content") else response
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        texts = []
        for item in raw:
            if isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            return "\n".join(texts)
    return str(raw)

def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for API transmission."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64

def _parse_image_response(response_text: str) -> tuple:
    """
    Parse structured response from vision model.
    Expected format: answer text with optional [CITE: page X] markers
    """
    citations = []
    clean_text = response_text
    
    # Extract citations in format [CITE: page X] or [CITE: halaman X]
    citation_pattern = r'\[CITE:\s*(?:halaman|page)?\s*(\d+)\]'
    import re
    matches = re.finditer(citation_pattern, response_text, re.IGNORECASE)
    
    for match in matches:
        page_num = int(match.group(1))
        citations.append(Citation(
            page=page_num,
            filename="[Image]",  # Images don't have filenames
            snippet="[Lihat gambar halaman ini]"
        ))
    
    # Remove citation markers from text
    clean_text = re.sub(citation_pattern, '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text, citations

def image_researcher_node(state: GraphState) -> GraphState:
    """
    Image Researcher Node: Analyze images via Gemini multimodal.
    
    Called when query_intent is 'image' or 'mixed' and image_pages are specified.
    Uses Gemini 3.1 Flash Lite for vision analysis.
    
    Args:
        state: GraphState containing query, image_pages, and pdf_pages
    
    Returns:
        Updated state with image_answer and image_citations
    """
    query = state["query"]
    image_pages = state.get("image_pages", [])
    pdf_pages = state.get("pdf_pages", {})
    
    # If no image pages specified, or no rendered pages available, return a helpful message.
    if not image_pages:
        return {
            **state,
            "image_answer": "Saya butuh nomor halaman gambar yang dimaksud (mis. 'gambar di halaman 2').",
            "image_citations": []
        }

    if not pdf_pages:
        return {
            **state,
            "image_answer": "Gambar belum tersedia untuk dianalisis (halaman belum dirender menjadi image). Silakan coba lagi setelah dokumen termuat sepenuhnya.",
            "image_citations": []
        }
    
    # Filter to only pages that exist in pdf_pages
    available_pages = [p for p in image_pages if p in pdf_pages]
    if not available_pages:
        return {
            **state,
            "image_answer": "Halaman yang Anda tanyakan tidak tersedia sebagai gambar.",
            "image_citations": []
        }
    
    try:
        # Get vision LLM (Gemini multimodal)
        llm_vision = get_llm_vision_gemini()
        
        # Prepare images for API
        images_b64 = []
        for page_num in available_pages:
            img = pdf_pages[page_num]
            b64 = _image_to_base64(img)
            images_b64.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}"
                }
            })
        
        # Build message with images
        system_prompt = (
            "Anda adalah analis visual untuk dokumen PDF. Jawab ringkas, rapi, dan mudah dibaca.\n\n"
            "Aturan format WAJIB:\n"
            "- Output SELALU 3 bagian:\n"
            "  1) Ringkasan (2-3 kalimat)\n"
            "  2) Poin penting (DAFTAR BERNOMOR)\n"
            "  3) Kesimpulan (1 kalimat)\n"
            "- Jangan gunakan heading markdown seperti ### atau subjudul panjang.\n"
            "- Jangan menulis paragraf panjang. Pecah jadi kalimat-kalimat pendek.\n"
            "- Jika ada simbol seperti ⊕/⊗, jelaskan dengan teks (mis. 'skip connection'), jangan spam simbolnya.\n"
            "- Untuk klaim yang didukung visual, tambahkan [CITE: halaman X] di akhir kalimat.\n"
            "- PENTING: Jangan gunakan bullet seperti '*' atau '-' untuk poin penting. Gunakan format: '1) ...', '2) ...'.\n"
        )
        
        # Build content with images and text
        content = images_b64 + [
            {
                "type": "text",
                "text": f"Pertanyaan: {query}\n\nAnalisis gambar ini dan jawab pertanyaan dengan detail."
            }
        ]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]
        
        # Call vision model
        response = llm_vision.invoke(messages)

        # Extract response text (Gemini often returns list content)
        response_text = _extract_text_from_response(response)
        
        # Parse citations from response
        clean_answer, citations = _parse_image_response(response_text)
        
        # Ensure all citations reference the correct page numbers
        for citation in citations:
            if citation["page"] not in available_pages:
                citation["page"] = available_pages[0]  # Fallback to first available page
        
        return {
            **state,
            "image_answer": clean_answer,
            "image_citations": citations
        }
    
    except Exception as e:
        error_msg = (
            f"Maaf, terjadi kesalahan saat menganalisis gambar. "
            f"Error: {str(e)[:200]}"
        )
        return {
            **state,
            "image_answer": error_msg,
            "image_citations": []
        }