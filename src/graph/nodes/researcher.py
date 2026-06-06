import json
import re
import concurrent.futures
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import TOP_K_RESULTS, get_llm
from src.graph.state import GraphState, Citation


def _extract_text_from_response(response) -> str:
    """
    Extract plain text from Gemini response reliably.
    
    Gemini via LangChain can return response in various formats:
    - String: langsung return
    - List of dicts: [{'type': 'text', 'text': '...'}, ...]
    - AIMessage with .content as list
    """
    # Try .text first (Gemini native)
    if hasattr(response, 'text') and response.text:
        return response.text
    
    raw = response.content if hasattr(response, 'content') else response
    
    # If it's already a string
    if isinstance(raw, str):
        return raw
    
    # If it's a list (Gemini candidates format)
    if isinstance(raw, list):
        texts = []
        for item in raw:
            if isinstance(item, dict):
                # Standard Gemini format: {"type": "text", "text": "..."}
                if 'text' in item:
                    texts.append(item['text'])
                # Fallback: try to stringify safely
                elif 'type' in item and item['type'] == 'text':
                    texts.append(str(item.get('text', '')))
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            return "\n".join(texts)
    
    # Last resort
    return str(raw)


def _clean_markdown_tables(text: str) -> str:
    """
    Clean and normalize markdown tables for consistent Streamlit rendering.
    
    Fixes:
    - Removes extra blank lines before/after tables
    - Ensures consistent spacing in pipe-delimited rows
    - Removes duplicate header separators
    - Normalizes alignment indicators (|:---|:---|)
    """
    lines = text.split('\n')
    cleaned_lines = []
    in_table = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect table rows (start with |)
        if stripped.startswith('|') and stripped.endswith('|'):
            in_table = True
            
            # Skip if line is only alignment separators (|:---|---|)
            if re.match(r'^\|\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|$', stripped):
                # Only add separator if previous line exists and isn't empty
                if cleaned_lines and cleaned_lines[-1].strip():
                    cleaned_lines.append(stripped)
            else:
                # Regular table row - normalize spacing
                cells = [cell.strip() for cell in stripped.split('|')[1:-1]]
                normalized_row = '| ' + ' | '.join(cells) + ' |'
                cleaned_lines.append(normalized_row)
        else:
            # Non-table line
            if in_table and stripped == '':
                # Skip blank lines immediately after tables
                continue
            in_table = False
            if stripped or (cleaned_lines and cleaned_lines[-1].strip()):
                cleaned_lines.append(line)
    
    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)


def _parse_citations(text: str) -> tuple:
    """
    Parse structured citations from the LLM response.
    
    Extracts JSON citation blocks and returns clean answer text + citation list.
    Handles markdown tables and normalizes formatting.
    """
    citations = []
    clean_text = text
    
    # Try to extract JSON citation blocks (both with and without markdown formatting)
    # First, handle ```json ... ``` blocks
    json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    
    # We need to process blocks one by one
    for match in re.finditer(json_block_pattern, clean_text, re.DOTALL | re.IGNORECASE):
        full_block = match.group(0)
        json_str = match.group(1)
        try:
            citation = json.loads(json_str)
            citations.append(Citation(
                page=citation.get("page"),
                filename=citation.get("filename"),
                snippet=citation.get("snippet")
            ))
            # Remove the full markdown block
            clean_text = clean_text.replace(full_block, "")
        except (json.JSONDecodeError, AttributeError):
            pass

    # Then, handle bare JSON blocks (even if page/filename are omitted by the LLM)
    citation_pattern = r'\{[^{}]*"snippet"[^{}]*\}'
    matches = re.findall(citation_pattern, clean_text)
    
    for match in matches:
        try:
            citation = json.loads(match)
            if "snippet" in citation:
                citations.append(Citation(
                    page=citation.get("page"),
                    filename=citation.get("filename"),
                    snippet=citation.get("snippet")
                ))
                clean_text = clean_text.replace(match, "", 1)
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Clean up markdown tables
    clean_text = _clean_markdown_tables(clean_text)
    
    # Clean up extra whitespace (but preserve newlines for tables)
    clean_text = re.sub(r' +', ' ', clean_text)  # Multiple spaces → single space
    clean_text = re.sub(r'\n\n\n+', '\n\n', clean_text)  # Multiple newlines → double newline
    clean_text = clean_text.strip()
    
    return clean_text, citations


def research_node(state: GraphState, vector_store) -> GraphState:
    """
    LangGraph node: Researcher
    
    Takes the user query (or expanded multi-queries), retrieves relevant
    context from FAISS using multi-query search, and generates a draft
    answer with citations using the selected LLM (Gemini or Groq).
    
    If query_variations is present in state (from Query Rewriter), uses
    multi_query_search for broader recall. Otherwise falls back to
    single similarity_search.
    
    Args:
        state: Current graph state with query + selected_model
        vector_store: FAISSDocumentStore instance for search
    
    Returns:
        Updated state with draft_answer, draft_citations, and context_chunks
    """
    query = state["query"]
    selected_model = state.get("selected_model", None)
    query_variations = state.get("query_variations", None)
    needs_synthesis = state.get("needs_synthesis", False)
    sub_queries = state.get("sub_queries", [])
    
    # ── Multi-Hop Parallel Processing ────────────────────────────
    if needs_synthesis and sub_queries:
        print(f"\n[DEBUG] Memproses {len(sub_queries)} sub-queries secara paralel...")
        llm = get_llm(model_name=selected_model)
        
        def process_sub_query(sq: str) -> dict:
            # 1. Search
            chunks = vector_store.multi_query_search([sq], k=15, exclude_content_type="image")
            
            # 2. Build context
            ctx_parts = []
            for i, chunk in enumerate(chunks):
                meta = chunk["metadata"]
                ctx_parts.append(
                    f"[Konteks {i+1} | File: {meta['filename']} | Halaman: {meta['page_num']}]\n"
                    f"{chunk['chunk_text']}\n"
                )
            ctx_str = "\n".join(ctx_parts)
            
            # 3. Prompt
            sys_prompt = (
                "Anda adalah asisten analis finansial yang menjawab pertanyaan spesifik berdasarkan konteks.\n"
                "ATURAN:\n"
                "1. Jawab berdasarkan konteks dokumen yang diberikan secara ringkas namun akurat.\n"
                "2. Jika informasi tidak ada dalam konteks, katakan 'Tidak ditemukan'.\n"
                "3. WAJIB Sertakan kutipan dalam format blok kode JSON (```json ... ```):\n"
                "   ```json\n   {\"page\": <hal>, \"filename\": \"<file>\", \"snippet\": \"<teks>\"}\n   ```\n"
            )
            usr_prompt = f"**Konteks:**\n{ctx_str}\n\n**Pertanyaan:** {sq}"
            
            # 4. Invoke LLM
            messages = [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=usr_prompt)
            ]
            try:
                response = llm.invoke(messages)
                raw_ans = _extract_text_from_response(response)
                clean_ans, cits = _parse_citations(raw_ans)
            except Exception as e:
                clean_ans = f"(Gagal memproses sub-query: {str(e)[:100]})"
                cits = []
                
            return {
                "sub_query": sq,
                "answer": clean_ans,
                "citations": cits,
                "chunks": chunks
            }
            
        sub_answers = []
        all_chunks = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_sq = {executor.submit(process_sub_query, sq): sq for sq in sub_queries}
            for future in concurrent.futures.as_completed(future_to_sq):
                res = future.result()
                sub_answers.append(res)
                all_chunks.extend(res["chunks"])
                
        # Deduplicate chunks for state
        seen = set()
        deduped_chunks = []
        for ch in all_chunks:
            meta = ch.get("metadata", {})
            parent_id = meta.get("parent_id")
            if parent_id is not None:
                key = parent_id
            else:
                key = (meta.get("filename"), meta.get("page_num"), meta.get("chunk_index"))
                
            if key not in seen:
                seen.add(key)
                if "parent_text" in meta:
                    ch["chunk_text"] = meta["parent_text"]
                deduped_chunks.append(ch)
                
        return {
            **state,
            "sub_answers": sub_answers,
            "context_chunks": deduped_chunks[:TOP_K_RESULTS + 10],
            "text_answer": None,
            "text_citations": None
        }

    # Detect simple metadata queries where the answer is typically on cover/title pages.
    # We force-include page 1 contexts to avoid missing obvious info like company name.
    q_lower = (query or "").lower()
    is_metadata_query = any(
        key in q_lower
        for key in [
            "perusahaan apa",
            "nama perusahaan",
            "ini dokumen tentang apa",
            "dokumen ini tentang apa",
            "dokumen apa",
            "laporan apa",
            "periode berapa",
            "tanggal berapa",
            "as of",
        ]
    )
    
    # ── Multi-Query Retrieval ─────────────────────────────────────
    # If query variations exist (from Query Rewriter), search each variation
    # and merge results for broader recall.
    if query_variations and len(query_variations) > 1:
        context_chunks = vector_store.multi_query_search(
            queries=query_variations,
            exclude_content_type="image"
        )
    else:
        # Fallback: single query search
        context_chunks = vector_store.multi_query_search(
            queries=[query],
            exclude_content_type="image"
        )

    # Detect simple metadata queries where the answer is typically on cover/title pages.
    # We force-include page 1 contexts to avoid missing obvious info like company name.
    is_financial_query = any(
        key in q_lower
        for key in [
            "aset", "liabilitas", "ekuitas", "laba", "rugi", "arus kas",
            "pendapatan", "beban", "asset", "liability", "equity", "income", "loss", "cash flow"
        ]
    )
    
    if is_metadata_query or is_financial_query:
        cover_chunks = []
        
        # Dynamically identify financial pages
        financial_pages = set()
        if is_financial_query:
            fin_keywords = [
                "laporan posisi keuangan", "statements of financial position", "balance sheet",
                "laporan laba rugi", "statements of profit or loss", "income statement",
                "laporan arus kas", "statements of cash flows",
                "laporan perubahan ekuitas", "statements of changes in equity"
            ]
            for text, meta in zip(getattr(vector_store, "text_store", []), getattr(vector_store, "metadata_store", [])):
                try:
                    text_lower = text.lower()
                    if any(fk in text_lower for fk in fin_keywords):
                        page_num = int(meta.get("page_num", -1))
                        if page_num != -1:
                            financial_pages.add(page_num)
                except Exception:
                    continue
                    
        for text, meta in zip(getattr(vector_store, "text_store", []), getattr(vector_store, "metadata_store", [])):
            try:
                page_num = int(meta.get("page_num", -1))
                if is_metadata_query and page_num == 1:
                    cover_chunks.append({"chunk_text": text, "score": -1.0, "metadata": meta})
                elif is_financial_query and page_num in financial_pages:
                    # For financial queries, prioritize parent chunks (level == 'parent' or no level) to get full tables
                    if meta.get("level", "parent") == "parent":
                        cover_chunks.append({"chunk_text": text, "score": -1.0, "metadata": meta})
            except Exception:
                continue

        # Prepend cover chunks and deduplicate by parent_id if available, otherwise by (filename,page,chunk_index)
        merged = cover_chunks + context_chunks
        seen = set()
        deduped = []
        for ch in merged:
            m = ch.get("metadata", {})
            # Use parent_id for deduplication if it's a child chunk, else use the normal key
            parent_id = m.get("parent_id")
            if parent_id is not None:
                key = parent_id
            else:
                key = (m.get("filename"), m.get("page_num"), m.get("chunk_index"))
                
            if key in seen:
                continue
            seen.add(key)
            
            # If it's a child chunk, inject the parent text so the LLM has the full context (e.g. full table)
            if "parent_text" in m:
                ch["chunk_text"] = m["parent_text"]
                
            deduped.append(ch)
            
        # keep at most TOP_K_RESULTS + some cover extras (but cover is usually small)
        context_chunks = deduped[: max(TOP_K_RESULTS, 5) + 5]
    
    if not context_chunks:
        return {
            **state,
            "context_chunks": [],
            "text_answer": "Tidak ditemukan informasi relevan dalam dokumen yang diunggah.",
            "text_citations": [],
            "review_result": None,
            "final_answer": "Tidak ditemukan informasi relevan dalam dokumen yang diunggah.",
            "final_citations": []
        }
    
    # Build context string for the LLM
    context_parts = []
    print(f"\n[DEBUG] Number of retrieved chunks: {len(context_chunks)}")
    for i, chunk in enumerate(context_chunks):
        meta = chunk["metadata"]
        context_parts.append(
            f"[Konteks {i+1} | File: {meta['filename']} | Halaman: {meta['page_num']}]\n"
            f"{chunk['chunk_text']}\n"
        )
        print(f"\n--- Chunk {i+1} (Score: {chunk.get('score', 'N/A')}) ---")
        print(chunk.get("chunk_text", "")[:500] + "...")
        print("Metadata:", meta)
    context_str = "\n".join(context_parts)
    
    # Retrieve previous reviewer feedback if retrying
    feedback = ""
    if state.get("retry_count", 0) > 0 and state.get("review_result"):
        issues = state["review_result"].get("issues", [])
        if issues:
            feedback = (
                "\n\n**CATATAN REVISI dari Reviewer:**\n"
                + "\n".join(f"- {issue}" for issue in issues)
                + "\n\nPerbaiki jawaban Anda berdasarkan catatan di atas."
            )
    
    # Build prompt
    system_prompt = (
        "Anda adalah asisten AI analis finansial yang menjawab pertanyaan berdasarkan dokumen yang diberikan. "
        "Anda HARUS mematuhi aturan berikut:\n\n"
        "1. Jawablah HANYA berdasarkan konteks dokumen yang diberikan.\n"
        "2. Jika informasi tidak ada dalam konteks, katakan 'Tidak ditemukan dalam dokumen.'\n"
        "3. Sertakan kutipan dalam format blok kode JSON (```json ... ```) untuk setiap klaim yang Anda buat:\n"
        '   ```json\n   {"page": <nomor_halaman>, "filename": "<nama_file>", "snippet": "<teks_bukti_pendek>"}\n   ```\n'
        "4. Cantumkan kutipan blok kode JSON tersebut TEPAT setelah setiap kalimat/pernyataan yang didukung bukti.\n"
        "5. WAJIB MENGGUNAKAN BACKTICKS (```json) agar kutipan tidak bocor ke jawaban akhir pengguna.\n"
        "6. Jika menyebutkan angka, pastikan angka tersebut benar-benar ada di konteks.\n"
        "7. Jawab dalam Bahasa Indonesia yang baik dan benar.\n"
        "7. MARKDOWN TABLES: Jika teks mengandung [TABLE] atau data tabel, rekonstruksi menggunakan format markdown:\n"
        "   | Kolom 1 | Kolom 2 | Kolom 3 |\n"
        "   |---------|---------|----------|\n"
        "   | Data 1  | Data 2  | Data 3  |\n"
        "   Pastikan setiap baris dimulai dan diakhiri dengan pipe (|). Hitung jumlah baris, kolom, dan analisisnya.\n"
        "8. Format markdown harus rapi: spasi konsisten antar kolom, alignment sama rata.\n"
        "9. Jika pengguna menyebut 'tabel 2' / 'table 2', prioritaskan konteks yang mengandung tag [TABLE_2].\n"
        "   - Dalam dokumen ini: [TABLE_1] = Peneliti Terdahulu, [TABLE_2] = Arsitektur (Stage/Operator/Stride/Channels/Layers)."
    )
    
    user_prompt = (
        f"{feedback if feedback else ''}\n\n"
        f"**Konteks Dokumen:**\n{context_str}\n\n"
        f"**Pertanyaan:** {query}\n\n"
        f"**Instruksi:** Jawab pertanyaan berdasarkan konteks di atas. "
        f"Sertakan kutipan JSON untuk setiap klaim. "
        f"Jika konteks tidak cukup, katakan bahwa informasi tidak tersedia."
    )
    
    # Call Google Gemini LLM via factory
    llm = get_llm(model_name=selected_model)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        raw_answer = _extract_text_from_response(response)
        
        # Parse citations from the answer
        clean_answer, citations = _parse_citations(raw_answer)
        
    except Exception as e:
        # Fallback if LLM call fails
        clean_answer = (
            f"Maaf, terjadi kesalahan saat memproses pertanyaan. "
            f"Silakan coba lagi. Error: {str(e)[:200]}"
        )
        citations = []
    
    return {
        **state,
        "context_chunks": context_chunks,
        "text_answer": clean_answer,
        "text_citations": citations,
        "review_result": None,
        "final_answer": None,
        "final_citations": None
    }