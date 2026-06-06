"""
Enhanced Table Parser — Vision-Based (P0 Priority).

Strategy:
  1. Heuristic detection: check if a page likely contains tables using
     text-density heuristics. Skip non-table pages to conserve Gemini quota.
  2. Render page to image via pypdfium2 (zero native C++ deps, ARM-safe).
  3. Send image to Gemini 3.1 Flash Lite Vision via LangChain multimodal.
  4. Parse structured Markdown output, tag with metadata, return chunks
     compatible with the existing chunker.

No Camelot, no Ghostscript, no Tkinter — pure Python + Gemini Vision.
Works perfectly on Windows ARM (Snapdragon X).
"""
import io
import re
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

from src.config import (
    GOOGLE_API_KEY,
    RENDER_DPI,
    get_llm_vision_gemini,
)


# ─── Heuristic Table Detection ─────────────────────────────────────

_FINANCIAL_KEYWORDS = [
    "aset", "liabilitas", "ekuitas", "laba", "rugi", "beban",
    "pendapatan", "posisi keuangan", "laporan laba rugi",
    "calk", "catatan atas laporan keuangan", "neraca",
    "arus kas", "perubahan ekuitas", "modal", "saham",
    "pinjaman", "utang", "piutang", "persediaan", " kas ",
    "bank", "investasi", "properti", "akumulasi", "penyusutan",
    "amortisasi", "pajak", "npwp", "dividen", "saldo laba",
]

_FINANCIAL_NUM_PATTERN = re.compile(r'(?:Rp|USD|\$|EUR)?\s*[\d,]{3,}(?:\.\d+)?')


def _has_table_heuristics(page_text: str) -> Tuple[bool, float]:
    """
    Detect whether a page likely contains a table using lightweight heuristics.

    Returns:
        (has_table: bool, confidence: float 0.0-1.0)
    """
    if not page_text or len(page_text.strip()) < 30:
        return False, 0.0

    text = page_text.strip()
    lines = text.split('\n')
    words = text.split()

    if len(words) < 10:
        return False, 0.0

    signals = []
    total_weight = 0.0

    # Signal 1: High numeric density (>15% numeric tokens)
    numeric_count = sum(1 for w in words if re.match(r'^[\d,.\-Rp%()]+$', w))
    numeric_ratio = numeric_count / len(words) if words else 0
    if numeric_ratio > 0.15:
        signals.append(("high_numeric_density", numeric_ratio, 0.35))
        total_weight += 0.35

    # Signal 2: Lines with 3+ whitespace-separated tokens (tabular layout)
    multi_token_lines = sum(1 for l in lines if len(l.split()) >= 4)
    if len(lines) > 0 and multi_token_lines / len(lines) > 0.4:
        signals.append(("multi_token_lines", multi_token_lines / len(lines), 0.20))
        total_weight += 0.20

    # Signal 3: Double-space/tab separators OR markdown table pipes (|)
    tabular_lines = sum(1 for l in lines if re.search(r'  {2,}', l) or '\t' in l or '|' in l)
    if len(lines) > 0 and tabular_lines / len(lines) > 0.15:
        signals.append(("tabular_lines", tabular_lines / len(lines), 0.25))
        total_weight += 0.25

    # Signal 4: Financial section keywords
    keyword_matches = sum(1 for kw in _FINANCIAL_KEYWORDS if kw in text.lower())
    if keyword_matches >= 2:
        kw_score = min(keyword_matches / 5.0, 1.0)
        signals.append(("financial_keywords", kw_score, 0.20))
        total_weight += 0.20

    # Signal 5: Financial number pattern (Rp XXX, "1.234.567", etc.)
    num_matches = len(_FINANCIAL_NUM_PATTERN.findall(text))
    if num_matches >= 3:
        num_score = min(num_matches / 10.0, 1.0)
        signals.append(("financial_numbers", num_score, 0.10))
        total_weight += 0.10

    # Compute weighted confidence
    if total_weight == 0:
        return False, 0.0

    confidence = sum(
        score * weight for _, score, weight in signals
    ) / total_weight

    # Threshold: need at least 2 signals and confidence > 0.3
    has_table = len(signals) >= 2 and confidence > 0.3

    return has_table, confidence


# ─── Page Rendering ────────────────────────────────────────────────

def _render_page_to_base64(file_bytes: bytes, page_index: int, dpi: int = None) -> Optional[str]:
    """
    Render a PDF page as a base64-encoded PNG image.

    Uses pypdfium2 (already installed, ARM-compatible, no native deps).

    Args:
        file_bytes: Raw PDF file bytes.
        page_index: 0-based page index.
        dpi: Rendering DPI (default: RENDER_DPI from config).

    Returns:
        Base64-encoded PNG string, or None on failure.
    """
    import pypdfium2 as pdfium

    if dpi is None:
        dpi = RENDER_DPI

    try:
        pdf = pdfium.PdfDocument(io.BytesIO(file_bytes))

        if page_index >= len(pdf):
            pdf.close()
            return None

        page = pdf[page_index]
        bitmap = page.render(scale=dpi / 72)
        pil_img = bitmap.to_pil()

        # Encode to base64 PNG
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        page.close()
        pdf.close()

        return b64_str

    except Exception as e:
        print(f"  ⚠️ Render halaman {page_index + 1} gagal: {e}")
        return None


# ─── Gemini Vision Prompt ──────────────────────────────────────────

TABLE_EXTRACTION_SYSTEM_PROMPT = """Anda adalah asisten ekstraktor tabel laporan keuangan Indonesia. Tugas Anda: ekstrak SEMUA tabel yang terlihat pada gambar halaman PDF ini dan keluarkan dalam format Markdown terstruktur.

ATURAN KETAT:
1. Ekstrak SEMUA tabel — termasuk tabel multi-level header, tabel di CALK, dan tabel kecil.
2. Header multi-level: jika tabel memiliki header bertingkat (contoh "31 Maret 2025" di atas "Audited" / "Unaudited"), gunakan dua baris header dalam satu tabel.
3. ANGKA: Ekstrak PERSIS seperti di gambar — jangan bulatkan, jangan ubah format. Termasuk tanda kurung untuk nilai negatif: (1.234) → -1.234.
4. MATA UANG: Pertahankan prefix "Rp" atau simbol.
5. Baris TOTAL / JUMLAH: Sertakan sebagai baris terakhir dengan label "[TOTAL]" atau "[JUMLAH]".
6. CATATAN KAKI: Jika ada footnote marker (*, **, 1), 2), huruf superskrip), tulis sebagai baris komentar setelah tabel.
7. Jika halaman TIDAK memiliki tabel sama sekali: keluarkan "[NO_TABLE_DETECTED]"
8. Jika tabel terpotong (bersambung ke halaman berikutnya): catat "[TABLE_CONTINUED]" di bagian atas.
9. BARIS DATA: Hitung dan cantumkan jumlah baris data di komentar.

FORMAT OUTPUT MARKDOWN:
```markdown
<!-- TABLE | Halaman [N] | [Judul/Deskripsi singkat] -->
| Kolom Header 1 | Kolom Header 2 | Kolom Header 3 |
|----------------|----------------|----------------|
| Data 1         | Data 2         | Data 3         |
| Data 4         | Data 5         | Data 6         |
<!-- Baris data: [X] baris -->
<!-- CATATAN KAKI: [teks footnote jika ada] -->
```"""


# ─── Markdown Parsing ──────────────────────────────────────────────

def _parse_markdown_tables(markdown_text: str) -> List[str]:
    """
    Extract stand-alone Markdown tables from Gemini's response.

    Each table starts and ends with ```markdown ... ``` fences.
    If no fences are found, fallback: treat the whole text as a single table block.

    Returns:
        List of table Markdown strings (without fences).
    """
    if not markdown_text:
        return []

    # Try fenced code blocks first
    pattern = r'```(?:markdown)?\s*\n(.*?)```'
    matches = re.findall(pattern, markdown_text, re.DOTALL)

    if matches:
        return [m.strip() for m in matches if m.strip()]

    # Fallback: entire response is a table
    # Only use if it contains pipe characters (table indicator)
    if '|' in markdown_text:
        return [markdown_text.strip()]

    return []


def _parse_table_metadata(table_md: str, page_num: int, filename: str, table_index: int) -> Dict:
    """
    Extract metadata from a parsed Markdown table.

    Looks for HTML comments like:
      <!-- TABLE | Halaman N | Deskripsi -->
      <!-- Baris data: X -->
      <!-- CATATAN KAKI: ... -->

    Args:
        table_md: Raw Markdown string of a single table.
        page_num: Page number (1-based).
        filename: Source PDF filename.
        table_index: Global sequential table index.

    Returns:
        dict with keys: text (cleaned), metadata dict.
    """
    description = ""
    row_count = 0
    footnotes = ""

    # Extract HTML comments
    comments = re.findall(r'<!--(.*?)-->', table_md, re.DOTALL)
    for comment in comments:
        comment = comment.strip()
        if comment.startswith("TABLE |"):
            parts = [p.strip() for p in comment.split("|")]
            if len(parts) >= 3:
                description = parts[2]
        elif comment.startswith("Baris data"):
            match = re.search(r'(\d+)', comment)
            if match:
                row_count = int(match.group(1))
        elif comment.startswith("CATATAN KAKI"):
            footnotes = comment.split(":", 1)[-1].strip() if ":" in comment else ""
        elif comment.startswith("TABLE_CONTINUED"):
            description = "(BERSAMBUNG) " + description

    # Clean comments from table text
    clean_text = re.sub(r'<!--.*?-->', '', table_md, flags=re.DOTALL).strip()
    # Clean markdown code fences
    clean_text = re.sub(r'```markdown\s*', '', clean_text)
    clean_text = re.sub(r'```', '', clean_text).strip()

    # Build the final text content
    header_tag = f"[TABLE_{table_index}]"
    if description:
        header_tag += f" [{description}]"
    if footnotes:
        header_tag += f"\n<!-- Footnotes: {footnotes} -->"

    full_text = f"{header_tag}\n{clean_text}"

    metadata = {
        "filename": filename,
        "page_num": page_num,
        "table_index": table_index,
        "content_type": "table",
        "table_description": description,
        "table_row_count": row_count,
        "footnotes": footnotes,
    }

    return {"text": full_text, "metadata": metadata}


# ─── Main Entry Point ──────────────────────────────────────────────

def extract_tables_from_page(
    file_bytes: bytes,
    page_text: str,
    page_num: int,
    page_index: int,
    filename: str,
    global_table_counter: List[int],
    counter_lock: threading.Lock,
    force: bool = False,
) -> List[Dict]:
    """
    Extract tables from a single PDF page using Gemini Vision.

    Pipeline:
      1. Heuristic detection (skip if unlikely to have tables, unless forced).
      2. Render page to base64 PNG.
      3. Send to Gemini 3.1 Flash Lite Vision.
      4. Parse Markdown output → table chunks.
      5. Tag with [TABLE_N] metadata.

    Args:
        file_bytes: Raw PDF bytes.
        page_text: Text extracted by PyPDF2 (for heuristic check).
        page_num: 1-based page number.
        page_index: 0-based page index for rendering.
        filename: Source PDF filename.
        global_table_counter: Mutable list with a single int [count] for sequential IDs.
        counter_lock: Lock to make counter increments thread-safe.
        force: If True, skip heuristic check and always call Vision.

    Returns:
        List of chunk dicts compatible with chunker.chunk_documents():
        [{text, metadata: {filename, page_num, table_index, ...}}]
    """
    # Step 1: Heuristic detection
    if not force:
        has_table, confidence = _has_table_heuristics(page_text)
        if not has_table:
            # print(f"  ⏭️ Halaman {page_num}: skip Vision (tabel tidak terdeteksi, confidence={confidence:.2f})")
            return []
    else:
        confidence = 1.0

    # Step 2: Render page to image
    b64_image = _render_page_to_base64(file_bytes, page_index, dpi=200)
    if b64_image is None:
        return []

    # Step 3: Send to Gemini Vision
    if not GOOGLE_API_KEY:
        print(f"  ⚠️ Halaman {page_num}: GOOGLE_API_KEY tidak disetel — lewati ekstraksi tabel")
        return []

    llm = get_llm_vision_gemini(max_tokens=8192)

    from langchain_core.messages import HumanMessage

    msg = HumanMessage(content=[
        {"type": "text", "text": TABLE_EXTRACTION_SYSTEM_PROMPT},
        {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{b64_image}",
        },
    ])

    try:
        response = llm.invoke([msg])
        raw_text = response.text if hasattr(response, "text") else str(response.content)
    except Exception as e:
        print(f"  ❌ Halaman {page_num}: Gemini Vision gagal — {e}")
        return []

    if not raw_text or "[NO_TABLE_DETECTED]" in raw_text:
        # print(f"  ⏭️ Halaman {page_num}: Gemini mengonfirmasi tidak ada tabel")
        return []

    # Step 4: Parse Markdown tables
    table_blocks = _parse_markdown_tables(raw_text)
    if not table_blocks:
        return []

    # Step 5: Tag each table with metadata
    chunks = []
    for table_md in table_blocks:
        with counter_lock:
            global_table_counter[0] += 1
            table_index = global_table_counter[0]

        parsed = _parse_table_metadata(table_md, page_num, filename, table_index)
        chunks.append(parsed)

    print(f"  📊 Halaman {page_num}: {len(chunks)} tabel diekstrak via Gemini Vision (confidence={confidence:.2f})")

    return chunks


# ─── Batch Processing ──────────────────────────────────────────────

def extract_all_tables(
    file_bytes: bytes,
    pages_data: List[Dict],
    filename: str,
    force_all: bool = False,
) -> List[Dict]:
    """
    Process all pages in a document for table extraction.

    This function is designed to be called from pdf_loader.extract_pdf_text()
    or main.py after the initial PyPDF2 extraction.

    Args:
        file_bytes: Raw PDF bytes.
        pages_data: List of {page_num, text, filename} from pdf_loader.
        filename: Source PDF filename.
        force_all: If True, check ALL pages (even text-heavy ones).

    Returns:
        List of table chunk dicts (flat list, all pages).
    """
    global_counter = [0]  # Mutable counter for sequential table IDs
    counter_lock = threading.Lock()
    all_table_chunks = []
    
    # Store futures to preserve order or track completion
    futures_map = {}

    print(f"  ⚡ Memulai pemrosesan paralel dengan max_workers=1...")
    with ThreadPoolExecutor(max_workers=1) as executor:
        for idx, page in enumerate(pages_data):
            page_text = page.get("text", "")
            page_num = page.get("page_num", idx + 1)

            future = executor.submit(
                extract_tables_from_page,
                file_bytes=file_bytes,
                page_text=page_text,
                page_num=page_num,
                page_index=idx,
                filename=filename,
                global_table_counter=global_counter,
                counter_lock=counter_lock,
                force=force_all,
            )
            futures_map[future] = page_num
            
        for future in as_completed(futures_map):
            page_num = futures_map[future]
            try:
                chunks = future.result()
                if chunks:
                    all_table_chunks.extend(chunks)
            except Exception as e:
                print(f"  ❌ Gagal memproses halaman {page_num}: {e}")

    # Urutkan ulang chunk berdasarkan page_num agar berurutan saat di-embed
    all_table_chunks.sort(key=lambda x: x["metadata"]["page_num"])

    return all_table_chunks