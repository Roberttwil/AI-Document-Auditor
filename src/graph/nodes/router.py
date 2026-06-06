import re
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import get_llm
from src.graph.state import GraphState


def _extract_page_numbers(text: str) -> List[int]:
    """
    Extract page numbers from text.
    Handles various formats: "halaman 2", "page 2", "hal. 2", "hlm 2", etc.
    """
    # Regex patterns untuk berbagai format
    patterns = [
        r'(?:halaman|hal\.?|hlm|page|p\.?)\s*(\d+)',
        r'(?:di\s+)?halaman\s+ke-?(\d+)',
        r'gambar\s+(?:di\s+)?halaman\s+(\d+)',
        r'chart\s+(?:di\s+)?halaman\s+(\d+)',
        r'grafik\s+(?:di\s+)?halaman\s+(\d+)',
    ]
    
    pages = set()
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        pages.update(int(m) for m in matches)
    
    return sorted(list(pages))


def _classify_intent(query: str, llm_response: str) -> str:
    """
    Parse LLM response to extract intent classification.
    LLM returns JSON with intent field.
    """
    try:
        # Try to extract JSON from response
        import json
        # Look for JSON structure in response
        json_match = re.search(r'\{[^{}]*"intent"[^{}]*\}', llm_response)
        if json_match:
            data = json.loads(json_match.group())
            return data.get('intent', 'text').lower()
    except:
        pass
    
    # Fallback: keyword-based heuristic
    query_lower = query.lower()

    image_keywords = [
        "gambar", "grafik", "chart", "diagram", "visualisasi",
        "plot", "image", "picture",
    ]
    text_structured_keywords = [
        "tabel", "table", "kolom", "baris", "akurasi", "dataset", "ringkasan"
    ]
    mixed_keywords = ["bandingkan", "vs", "versus", "perbedaan", "persamaan"]

    has_image = any(kw in query_lower for kw in image_keywords)
    has_text = any(kw in query_lower for kw in text_structured_keywords)
    wants_compare = any(kw in query_lower for kw in mixed_keywords)

    # If query mentions both table/text and image, treat as mixed
    if has_image and has_text:
        return "mixed"

    # If user explicitly wants comparison and mentions image, treat as mixed
    if wants_compare and has_image:
        return "mixed"

    if has_image:
        return "image"

    return "text"


def router_node(state: GraphState) -> GraphState:
    """
    Router Node: Detect query intent (text/image/mixed) and extract page references.
    
    Uses LLM to intelligently classify:
    - 'text': Pertanyaan tentang konten teks (tabel, angka, deskripsi)
    - 'image': Pertanyaan tentang gambar/grafik/diagram
    - 'mixed': Pertanyaan yang menggabungkan keduanya
    
    Args:
        state: GraphState dengan field 'query'
    
    Returns:
        Updated state dengan 'query_intent' dan 'image_pages'
    """
    query = state["query"]
    selected_model = state.get("selected_model", None)
    
    # Build classification prompt
    system_prompt = (
        "Anda adalah router AI yang mengklasifikasi jenis pertanyaan pengguna. "
        "Tentukan apakah pertanyaan berkaitan dengan:\n"
        "- 'text': Data teks, angka, tabel, deskripsi\n"
        "- 'image': Gambar, grafik, diagram, chart, visualisasi\n"
        "- 'mixed': Kombinasi keduanya (misal: 'bandingkan tabel dengan grafik')\n\n"
        "Jika pertanyaan menyebut halaman tertentu, ekstrak nomor halaman.\n\n"
        "HANYA return JSON format:\n"
        '{"intent": "<text|image|mixed>", "pages": [<nomor halaman>], "reasoning": "<alasan singkat>"}'
    )
    
    user_prompt = f"Klasifikasikan pertanyaan ini:\n\n{query}"
    
    llm = get_llm(model_name=selected_model, tier="light")
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        
        # Extract response text
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Parse intent from LLM response
        intent = _classify_intent(query, response_text)
        
        # Extract page numbers
        pages = _extract_page_numbers(query)
        
    except Exception as e:
        # Fallback: simple heuristic if LLM fails
        intent = _classify_intent(query, "")
        pages = _extract_page_numbers(query)
    
    return {
        **state,
        "query_intent": intent,
        "image_pages": pages,
        "review_target": intent  # Will be updated after both researchers run
    }