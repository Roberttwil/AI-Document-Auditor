"""
Query Rewriter Node for Multi-Query Retrieval.

Generates multiple reformulations of the user's query to capture different
semantic angles, improving recall for the FAISS similarity search.

Example:
    User: "Bagaimana tren beban operasional?"
    → [
        "tren beban operasional tahun 2023",
        "perbandingan biaya usaha operasional per kuartal",
        "total beban operasional dari laporan laba rugi"
    ]
"""
from typing import List, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import MULTI_QUERY_COUNT, get_llm


def _extract_text_from_response(response) -> str:
    """Extract plain text from LLM response (works for Gemini + Groq)."""
    if hasattr(response, 'text') and response.text:
        return response.text
    raw = response.content if hasattr(response, 'content') else response
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        texts = []
        for item in raw:
            if isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            return "\n".join(texts)
    return str(raw)


def generate_query_variations(query: str, chat_history: List[Dict] = None, model_name: str = None) -> List[str]:
    """
    Generate multiple reformulations of the user query using LLM.
    
    The original query is always included as the first item.
    
    Args:
        query: Original user query
        chat_history: List of previous messages [{"role": "user", "content": "..."}, ...]
        model_name: Selected model display name (for LLM call)
    
    Returns:
        List of query strings [original, variation_1, variation_2, ...]
    """
    n_variations = max(0, MULTI_QUERY_COUNT - 1)  # -1 because original included
    
    if n_variations <= 0:
        return [query]
    
    # Use smallest/fastest model for query rewriting (Llama 3.1 8B via Groq if available,
    # or fallback to Gemini Flash Lite)
    rewrite_model = None
    try:
        # Try to use a lightweight model for rewriting
        from src.config import AVAILABLE_MODELS
        # Prefer Llama 3.1 8B (fastest) or any Groq model
        for candidate in ["🔬 Llama 3.1 8B (Groq)", "⚡ Gemini 3.1 Flash Lite (Default)"]:
            if candidate in AVAILABLE_MODELS:
                rewrite_model = candidate
                break
    except Exception:
        pass
    
    system_prompt = (
        "Anda adalah asisten AI ahli yang bertugas merakit ulang pertanyaan pengguna (Query Contextualization) "
        "agar menjadi jelas dan mandiri (standalone), serta membuat beberapa variasinya untuk pencarian dokumen.\n\n"
        "Aturan:\n"
        "1. Jika pertanyaan menggunakan kata ganti ('nya', 'itu', 'dia', 'perusahaan tersebut'), "
        "ganti kata tersebut dengan subjek aslinya berdasarkan Riwayat Percakapan.\n"
        "2. Variasi PERTAMA (Baris 1) HARUS merupakan pertanyaan utama yang sudah diperjelas konteksnya.\n"
        "3. Variasi selanjutnya adalah rephrasing/sinonim dari pertanyaan utama.\n"
        "4. Semua dalam Bahasa Indonesia.\n"
        "5. Jika pengguna bertanya tentang 'utang', pastikan untuk selalu menyertakan kata 'liabilitas' pada variasi yang dihasilkan.\n"
        "6. Jangan beri penjelasan. Keluarkan HANYA daftar variasi, satu per baris."
    )
    
    history_text = ""
    if chat_history:
        history_parts = []
        for msg in chat_history[-4:]: # Use last 4 messages to avoid token bloat
            role = "User" if msg.get("role") == "user" else "AI"
            history_parts.append(f"{role}: {msg.get('content')}")
        history_text = "Riwayat Percakapan:\n" + "\n".join(history_parts) + "\n\n"
        
    user_prompt = (
        f"{history_text}"
        f"Buatkan {n_variations + 1} variasi dari pertanyaan pengguna berikut (di mana baris ke-1 adalah pertanyaan yang sudah diperjelas subjeknya):\n\n"
        f"Pertanyaan Pengguna Saat Ini: {query}\n\n"
        f"Keluarkan {n_variations + 1} variasi, satu per baris, tanpa angka atau bullet."
    )
    
    llm = get_llm(model_name=rewrite_model, temperature=0.1, tier="light")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        raw = _extract_text_from_response(response)
        
        # Parse: split by newlines, strip, filter empty
        variations = []
        for line in raw.strip().split('\n'):
            line = line.strip().strip('-').strip('*').strip('"').strip("'").strip('0123456789.) ')
            if line and len(line) > 5:
                variations.append(line)
        
        # Deduplicate (case-insensitive)
        seen = set()
        deduped = []
        # If the LLM didn't return anything or failed, ensure we at least have the original query
        if not variations:
            variations = [query]
            
        for v in variations:
            key = v.lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(v)
        
        # Limit to MULTI_QUERY_COUNT
        result = deduped[:MULTI_QUERY_COUNT]
        
        # Ensure at least original query is returned
        if not result:
            result = [query]
        
        return result
        
    except Exception as e:
        # Fallback: just return the original query
        print(f"⚠️ Query rewriting failed: {e}")
        return [query]