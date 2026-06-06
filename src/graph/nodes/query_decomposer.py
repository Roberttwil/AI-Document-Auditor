"""
P1 — Query Decomposition Node for Multi-Hop Reasoning.

Breaks complex financial queries into atomic sub-queries, then synthesizes
answers from each sub-query into a coherent final response.

Example:
    Input: "Apakah rasio utang 56.65% masih di bawah batas maksimum 200%?"
    → Sub-queries:
        1. "Berapa rasio utang perusahaan?"
        2. "Berapa batas maksimum rasio utang menurut manajemen?"
        3. "Bandingkan rasio utang dengan batas maksimum."
"""
import json
import re
from typing import List, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import get_llm


def _extract_from_response(response) -> str:
    """Extract text from any LangChain response format."""
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


def decompose_query(query: str, model_name: str = None) -> List[str]:
    """
    Break a complex query into independent sub-queries.
    
    Returns a list of sub-query strings. If the query is already simple
    (doesn't need decomposition), returns [query].
    """
    system_prompt = (
        "Anda adalah asisten yang memecah pertanyaan kompleks tentang laporan keuangan "
        "menjadi sub-pertanyaan yang lebih sederhana dan independen.\n\n"
        "Aturan:\n"
        "1. Setiap sub-pertanyaan harus bisa dijawab secara independen dari sub-pertanyaan lain.\n"
        "2. Sub-pertanyaan harus MENCARI FAKTA, bukan membandingkan — simpan perbandingan "
        "untuk langkah sintesis terakhir.\n"
        "3. Gunakan Bahasa Indonesia.\n"
        "4. Contoh:\n"
        "   Input: 'Apakah rasio utang 56.65% masih di bawah batas maksimum 200%?'\n"
        "   Output:\n"
        "   - Berapa rasio utang perusahaan?\n"
        "   - Berapa batas maksimum rasio utang yang ditetapkan manajemen?\n"
        "5. Jangan beri penjelasan. Keluarkan HANYA daftar sub-pertanyaan, satu per baris, "
        "diawali dengan tanda '-' dan spasi."
    )
    
    user_prompt = (
        f"Pecah pertanyaan berikut menjadi sub-pertanyaan sederhana:\n\n"
        f"{query}\n\n"
        f"Keluarkan setiap sub-pertanyaan dalam baris terpisah diawali dengan '- '."
    )
    
    llm = get_llm(model_name=model_name, temperature=0.1, tier="light")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        raw = _extract_from_response(response)
        
        # Parse: lines starting with "- " or numbered
        sub_queries = []
        for line in raw.strip().split('\n'):
            line = line.strip()
            # Remove bullet markers: "- ", "1. ", "1) "
            cleaned = re.sub(r'^[\-\*\d]+[\.\)]\s*', '', line).strip()
            if cleaned and len(cleaned) > 5:
                sub_queries.append(cleaned)
        
        # Deduplicate
        seen = set()
        deduped = []
        for sq in sub_queries:
            key = sq.lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(sq)
        
        if deduped:
            return deduped
        
    except Exception as e:
        print(f"⚠️ Query decomposition failed: {e}")
    
    # Fallback: treat as single query
    return [query]


def synthesize_answers(
    query: str,
    sub_answers: List[Dict],
    model_name: str = None,
) -> str:
    """
    Synthesize answers from multiple sub-queries into a single coherent answer.
    
    Args:
        query: Original user query
        sub_answers: List of {sub_query, answer, citations} from each sub-query execution
        model_name: Selected LLM model
    
    Returns:
        Synthesized answer string with citations
    """
    if len(sub_answers) == 1:
        return sub_answers[0].get("answer", "")
    
    # Build context from all sub-answers
    context_parts = []
    for i, sa in enumerate(sub_answers):
        context_parts.append(
            f"[Sub-Query {i+1}]: {sa['sub_query']}\n"
            f"[Answer {i+1}]: {sa['answer']}\n"
        )
    context_str = "\n".join(context_parts)
    
    # Extract all citations
    all_citations = []
    for sa in sub_answers:
        all_citations.extend(sa.get("citations", []))
    
    system_prompt = (
        "Anda adalah asisten sintesis AI ahli keuangan. Tugas Anda: gabungkan jawaban dari beberapa "
        "sub-pertanyaan menjadi satu jawaban koheren yang langsung menjawab pertanyaan "
        "utama pengguna.\n\n"
        "Aturan:\n"
        "1. Jawab langsung pertanyaan UTAMA, jangan ulangi sub-pertanyaan.\n"
        "2. Jika sub-answer berisi angka/fakta yang bertentangan, identifikasi dan jelaskan.\n"
        "3. JIKA PERLU MELAKUKAN PERHITUNGAN MATEMATIKA (margin, selisih, rasio), ANDA WAJIB MENGGUNAKAN ALAT (TOOL) 'calculator'. Jangan pernah menebak hasil hitungan!\n"
        "4. Sertakan kutipan JSON untuk setiap klaim:\n"
        '   {"page": <nomor_halaman>, "filename": "<nama_file>", "snippet": "<teks_bukti>"}\n'
        "5. Jawab dalam Bahasa Indonesia yang baik dan benar.\n"
        "6. Jika informasi tidak cukup untuk menjawab pertanyaan utama, katakan dengan jujur."
    )
    
    user_prompt = (
        f"**Pertanyaan Utama:** {query}\n\n"
        f"**Jawaban dari Sub-Pertanyaan:**\n{context_str}\n\n"
        f"**Instruksi:** Gabungkan jawaban di atas menjadi satu jawaban koheren "
        f"yang langsung menjawab pertanyaan utama. Sertakan kutipan JSON. JIKA ADA PERHITUNGAN, GUNAKAN ALAT (TOOL)!"
    )
    
    llm = get_llm(model_name=model_name, temperature=0.1, tier="light")
    
    # Bind calculator tool
    from src.graph.tools import calculator
    from langchain_core.messages import ToolMessage
    
    llm_with_tools = llm.bind_tools([calculator])
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm_with_tools.invoke(messages)
        
        # Loop for tool calls (max 3 iterations to prevent infinite loop)
        iterations = 0
        while hasattr(response, "tool_calls") and response.tool_calls and iterations < 3:
            messages.append(response)
            
            for tool_call in response.tool_calls:
                if tool_call["name"] == "calculator":
                    expr = tool_call["args"].get("expression", "")
                    tool_result = calculator.invoke({"expression": expr})
                    
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ))
            
            # Re-invoke LLM with tool results
            response = llm_with_tools.invoke(messages)
            iterations += 1
            
        return _extract_from_response(response)
    except Exception as e:
        # Fallback: concatenate
        parts = [sa["answer"] for sa in sub_answers if sa.get("answer")]
        return "\n\n".join(parts) if parts else "(Sintesis gagal)"