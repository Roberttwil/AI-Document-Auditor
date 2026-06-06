import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import get_llm, GOOGLE_API_KEY, GROQ_API_KEY
from src.graph.state import GraphState, ReviewResult


# ─── Indonesian Financial Regulations for Legal Reasoning Check ───

# Keywords that signal the answer involves tax/policy/regulatory claims
_POLICY_KEYWORDS = [
    "pmk", "pp no", "psak", "oecd", "pilar 2", "pilar dua",
    "pajak", "tax", "peraturan", "undang-undang", "uu ",
    "kementerian", "menteri keuangan", "dirjen pajak",
    "wajib pajak", "npwp", "ppn", "pph", "pembukuan",
    "penyusutan", "amortisasi", "masa manfaat", "akuntansi",
    "psak 216", "psak 219", "psak 212", "psak 213",
    "konsolidasi", "entitas anak", "pengendalian",
    "nilai wajar", "biaya perolehan", "penurunan nilai",
]

# Map of known policy patterns to their expected treatments
# (used to catch common LLM hallucinations)
_POLICY_TREATMENT_RULES = {
    "oecd pilar 2": {
        "correct": "temporary mandatory exception",
        "wrong_patterns": ["permanent exception", "wajib diterapkan", "immediately effective", "diterapkan segera"],
        "explanation": "OECD Pillar 2 (Pilar 2) menerapkan 'temporary mandatory exception' — pengecualian sementara yang WAJIB, bukan opsional."
    },
    "psak 216": {
        "correct": "tanah tidak disusutkan",
        "wrong_patterns": ["tanah disusutkan", "penyusutan tanah"],
        "explanation": "PSAK 216 menyatakan tanah tidak disusutkan, hanya diturunkan nilainya jika ada indikasi penurunan nilai."
    },
    "pmk 136": {
        "correct": "temporary mandatory exception",
        "wrong_patterns": ["permanent exception", "dikecualikan permanen"],
        "explanation": "PMK 136/2024 mengadopsi OECD Pillar 2 dengan temporary mandatory exception."
    },
}


def _detect_policy_references(text: str) -> list:
    """Detect if the answer contains policy/regulatory references that need verification."""
    text_lower = text.lower()
    found = []
    for kw in _POLICY_KEYWORDS:
        if kw in text_lower:
            found.append(kw)
    return found


def _check_policy_treatment(answer: str, context_summary: str) -> list:
    """
    Check if the answer correctly applies known policy/regulatory treatments.
    
    For example:
    - OECD Pillar 2 → "temporary mandatory exception" (NOT "permanent" or "wajib diterapkan")
    - PSAK 216 → tanah tidak disusutkan
    - PMK 136/2024 → temporary mandatory exception
    
    Returns:
        List of issue strings, empty if no issues found.
    """
    answer_lower = answer.lower()
    context_lower = context_summary.lower() if context_summary else ""
    issues = []

    for policy, rule in _POLICY_TREATMENT_RULES.items():
        if policy not in answer_lower and policy not in context_lower:
            continue  # Policy not relevant to this answer

        # Check if the answer uses wrong treatment patterns
        for wrong in rule["wrong_patterns"]:
            if wrong in answer_lower:
                # Check if context actually supports this wrong treatment
                if wrong not in context_lower:
                    issues.append(
                        f"❌ Kesalahan interpretasi {policy.upper()}: jawaban menyebut '{wrong}', "
                        f"padahal yang benar adalah '{rule['correct']}'. "
                        f"{rule['explanation']}"
                    )

        # Check if the correct treatment is missing when it should be mentioned
        if rule["correct"] not in answer_lower and policy in context_lower:
            # This is a soft warning — only flag if context clearly mentions the rule
            if rule["correct"] in context_lower:
                issues.append(
                    f"⚠️ Kemungkinan tidak lengkap: {policy.upper()} disebut di konteks "
                    f"dengan '{rule['correct']}', tapi jawaban tidak mencantumkan istilah ini."
                )

    return issues


def _check_numerical_accuracy(answer: str, context_summary: str) -> list:
    """
    Check if all numbers in the answer can be found in the context.
    Extracts numbers from both and compares.
    """
    issues = []
    
    # Extract all numbers (including Rp, %, negative in parentheses)
    num_pattern = r'(?:Rp\s*)?[\d,]{2,}(?:\.\d+)?%?'
    answer_numbers = set(re.findall(r'[\d,]{2,}(?:\.\d+)?', answer))
    context_numbers = set(re.findall(r'[\d,]{2,}(?:\.\d+)?', context_summary))
    
    # Check for potential hallucinated numbers
    for num in answer_numbers:
        # Skip very short numbers (like years, page numbers)
        if len(num) <= 2 and num.isdigit():
            continue
        # Check if this number or a close variant exists in context
        found = False
        for ctx_num in context_numbers:
            if ctx_num in num or num in ctx_num:
                found = True
                break
        
        if not found:
            # This might be a hallucination — flag it
            issues.append(f"⚠️ Angka '{num}' dalam jawaban tidak ditemukan di konteks dokumen — kemungkinan halusinasi.")
    
    return issues


def review_node(state: GraphState) -> GraphState:
    """
    LangGraph node: Reviewer (Self-Correction for Dual-Agent)
    
    P3 — Enhanced with LEGAL REASONING checks for Indonesian financial regulations:
    1. Number hallucination: angka dalam jawaban ada di konteks?
    2. Page hallucination: nomor halaman yang dikutip sesuai metadata?
    3. Claim evidence: semua klaim didukung konteks?
    4. Policy misapplication: OECD Pillar 2, PSAK 216, PMK 136/2024, dll.
    5. Tax/regulatory treatment accuracy.
    6. Image visual accuracy.
    
    Args:
        state: Current graph state with text/image answers, context_chunks, and selected_model
    
    Returns:
        Updated state with review_result
    """
    review_target = state.get("review_target", "text")
    text_answer = state.get("text_answer", "")
    image_answer = state.get("image_answer", "")
    context_chunks = state.get("context_chunks", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    selected_model = state.get("selected_model", None)
    
    # Determine which answer(s) to review
    answer_to_review = ""
    if review_target == "text":
        answer_to_review = text_answer
    elif review_target == "image":
        answer_to_review = image_answer
    elif review_target == "mixed":
        if text_answer and image_answer:
            answer_to_review = f"[TEXT]\n{text_answer}\n\n[IMAGE]\n{image_answer}"
        elif text_answer:
            answer_to_review = text_answer
            review_target = "text"
        else:
            answer_to_review = image_answer
            review_target = "image"
    
    if not answer_to_review:
        return {
            **state,
            "review_result": ReviewResult(passed=True, issues=[]),
            "retry_count": retry_count,
            "final_answer": text_answer or image_answer
        }
    
    # Build context summary for reviewer
    context_summary = ""
    if review_target in ["text", "mixed"] and context_chunks:
        for i, chunk in enumerate(context_chunks):
            meta = chunk["metadata"]
            context_summary += (
                f"[Konteks {i+1} | File: {meta['filename']} | Halaman: {meta['page_num']}]\n"
                f"{chunk['chunk_text'][:400]}...\n\n"
            )
    
    # ── P3: Local rule-based checks (BEFORE LLM call) ──────────
    local_issues = []
    
    # Check 1: Numerical accuracy (fast, no LLM needed)
    local_issues.extend(_check_numerical_accuracy(answer_to_review, context_summary))
    
    # Check 2: Policy reference detection + legal treatment check
    policy_refs = _detect_policy_references(answer_to_review)
    if policy_refs:
        local_issues.extend(_check_policy_treatment(answer_to_review, context_summary))
    
    # ── P3: LLM-based review with enhanced legal reasoning prompt ──
    legal_context = ""
    if policy_refs:
        legal_context = (
            "\n**Perhatian Khusus — Regulasi Keuangan:**\n"
            "Jawaban mengandung referensi ke regulasi berikut: "
            + ", ".join(policy_refs) + ".\n"
            "Periksa dengan ketat apakah perlakuan akuntansi/peraturan sudah sesuai "
            "dengan ketentuan yang benar di konteks."
        )
    
    system_prompt = (
        "Anda adalah Reviewer AI KETAT untuk laporan keuangan Indonesia. "
        "Tugas Anda: periksa jawaban berdasarkan sumber dokumen yang tersedia.\n\n"
        "Periksa hal-hal berikut:\n"
        "1. **Halusinasi Angka**: Apakah semua angka/statistik dalam jawaban ADA di sumber?\n"
        "2. **Halusinasi Halaman**: Apakah nomor halaman yang disebutkan sesuai dengan metadata?\n"
        "3. **Klaim Tanpa Bukti**: Apakah ada klaim yang tidak didukung konteks?\n"
        "4. **Perlakuan Pajak/Regulasi**: Jika menyebut peraturan (PSAK, PMK, OECD), "
        "apakah perlakuan akuntansinya sudah benar?\n"
        "5. **Referensi CALK**: Jika menyebut Catatan Atas Laporan Keuangan, "
        "apakah nomor CALK sesuai dengan yang ada di konteks?\n"
        "6. **Konsistensi**: Jika ada jawaban teks dan gambar, apakah konsisten?\n"
        "7. **Akurasi Visual** (untuk gambar): Apakah interpretasi visual masuk akal?\n\n"
        f"{legal_context}\n\n"
        "Keluarkan hasil review dalam format JSON SAJA:\n"
        '{"passed": true/false, "issues": ["issue1", "issue2", ...]}'
    )
    
    context_section = f"**Konteks Dokumen (untuk validasi):**\n{context_summary}\n\n" if context_summary else ""
    user_prompt = (
        f"{context_section}"
        f"**Review Target**: {review_target}\n\n"
        f"**Jawaban yang Akan Direview:**\n{answer_to_review}\n\n"
        f"**Instruksi:** Review jawaban di atas. Jika ada masalah, "
        f"set passed=false dan sebutkan isu spesifik. Jika semua baik, set passed=true."
    )
    
    llm = get_llm(model_name=selected_model, temperature=0.0)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    llm_passed = True
    llm_issues = []
    
    try:
        response = llm.invoke(messages)
        review_text = response.content.strip()
        
        json_match = re.search(r'\{.*"passed".*\}', review_text, re.DOTALL)
        if json_match:
            review_data = json.loads(json_match.group())
            llm_passed = review_data.get("passed", True)
            llm_issues = review_data.get("issues", [])
        else:
            llm_passed = True
            llm_issues = []
            
    except Exception as e:
        llm_passed = True
        llm_issues = [f"⚠️ Review LLM error: {str(e)[:100]}"]
    
    # ── Merge local + LLM issues ─────────────────────────────
    all_issues = local_issues + llm_issues
    passed = llm_passed and len(local_issues) == 0
    
    # If only minor local warnings, still allow pass
    if local_issues and llm_passed:
        # Check if all local issues are warnings (start with ⚠️)
        all_warnings = all(i.startswith("⚠️") for i in local_issues)
        if all_warnings:
            passed = True  # Warnings don't block
    
    review_result = ReviewResult(passed=passed, issues=all_issues)
    new_retry_count = retry_count + 1 if not passed else retry_count
    
    if passed or new_retry_count >= max_retries:
        if review_target == "text":
            final_answer = text_answer
        elif review_target == "image":
            final_answer = image_answer
        else:
            parts = []
            if text_answer:
                parts.append(f"**Dari Dokumen Teks:**\n{text_answer}")
            if image_answer:
                parts.append(f"**Dari Analisis Gambar:**\n{image_answer}")
            final_answer = "\n\n".join(parts).strip() if parts else (text_answer or image_answer)
    else:
        final_answer = None
    
    return {
        **state,
        "review_target": review_target,
        "review_result": review_result,
        "retry_count": new_retry_count,
        "final_answer": final_answer
    }
