"""
Unit tests for the ingestion pipeline.
Tests the semantic chunker with heading detection and paragraph splitting.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion.chunker import chunk_documents, _is_heading_line, _split_by_headings


def test_chunker_empty_input():
    """Chunker should return empty list for empty input."""
    result = chunk_documents([])
    assert result == []


def test_chunker_single_short_page():
    """Chunker should handle a short page that fits in one chunk."""
    pages = [{
        "page_num": 1,
        "text": "Laporan keuangan menunjukkan laba bersih sebesar Rp 100 miliar.",
        "filename": "laporan_2024.pdf"
    }]
    
    chunks = chunk_documents(pages)
    
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["page_num"] == 1
    assert chunks[0]["metadata"]["filename"] == "laporan_2024.pdf"
    assert "laba bersih" in chunks[0]["text"]


def test_chunker_multiple_pages():
    """Chunker should handle multiple pages."""
    pages = [
        {
            "page_num": 1,
            "text": "Halaman pertama laporan keuangan.",
            "filename": "laporan.pdf"
        },
        {
            "page_num": 2,
            "text": "Halaman kedua berisi catatan atas laporan keuangan.",
            "filename": "laporan.pdf"
        }
    ]
    
    chunks = chunk_documents(pages)
    
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["page_num"] == 1
    assert chunks[1]["metadata"]["page_num"] == 2


def test_chunker_metadata_preservation():
    """Chunker should preserve filename and page_num in all chunks."""
    pages = [
        {
            "page_num": 5,
            "text": "Data penting ada di halaman 5.",
            "filename": "annual_report_2023.pdf"
        },
        {
            "page_num": 10,
            "text": "Lanjutan data di halaman 10.",
            "filename": "annual_report_2023.pdf"
        }
    ]
    
    chunks = chunk_documents(pages)
    
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["filename"] == "annual_report_2023.pdf"
    assert chunks[0]["metadata"]["page_num"] == 5
    assert chunks[1]["metadata"]["page_num"] == 10


def test_chunker_empty_page_text():
    """Chunker should skip empty page text."""
    pages = [
        {
            "page_num": 1,
            "text": "   ",
            "filename": "test.pdf"
        },
        {
            "page_num": 2,
            "text": "Halaman 2 berisi data.",
            "filename": "test.pdf"
        }
    ]
    
    chunks = chunk_documents(pages)
    
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["page_num"] == 2


def test_is_heading_line_bab():
    """BAB pattern should be detected as heading."""
    assert _is_heading_line("BAB I PENDAHULUAN")
    assert _is_heading_line("BAB II TINJAUAN PUSTAKA")
    assert _is_heading_line("BAB 1 PENDAHULUAN")
    assert _is_heading_line("BAB III METODE PENELITIAN")


def test_is_heading_letter_sub():
    """Letter sub-section patterns should be detected."""
    assert _is_heading_line("A. Latar Belakang")
    assert _is_heading_line("B. Rumusan Masalah")
    assert _is_heading_line("C. Tujuan Penelitian")


def test_is_heading_numbered():
    """Numbered patterns should be detected."""
    assert _is_heading_line("1. Pendahuluan")
    assert _is_heading_line("2. Landasan Teori")
    assert _is_heading_line("1.1 Latar Belakang")
    assert _is_heading_line("2.3.1 Analisis Data")


def test_is_heading_keyword():
    """Common Indonesian heading keywords should be detected."""
    assert _is_heading_line("Pendahuluan")
    assert _is_heading_line("Kesimpulan")
    assert _is_heading_line("Daftar Pustaka")
    assert _is_heading_line("Lampiran")


def test_is_heading_not():
    """Regular text should NOT be detected as heading."""
    assert not _is_heading_line("Ini adalah paragraf biasa.")
    assert not _is_heading_line("Angka laba bersih mencapai Rp 500 miliar.")
    assert not _is_heading_line("   ")
    assert not _is_heading_line("a")
    assert not _is_heading_line("laba bersih 12.34 miliar rupiah")


def test_split_by_headings_simple():
    """Text with headings should be split into sections."""
    text = """BAB I PENDAHULUAN
Ini adalah paragraf pendahuluan.

A. Latar Belakang
Latar belakang penelitian ini adalah...

B. Rumusan Masalah
Rumusan masalah dalam penelitian ini adalah..."""
    
    sections = _split_by_headings(text)
    assert len(sections) >= 3
    assert any("BAB I PENDAHULUAN" in s for s in sections)
    assert any("A. Latar Belakang" in s for s in sections)
    assert any("B. Rumusan Masalah" in s for s in sections)


def test_split_by_headings_no_heading():
    """Text without headings should remain as one section."""
    text = "Ini adalah paragraf biasa.\n\nIni paragraf lainnya.\n\nDan ini paragraf ketiga."
    sections = _split_by_headings(text)
    assert len(sections) == 1


def test_chunker_with_headings():
    """Full chunker should split by headings."""
    pages = [{
        "page_num": 1,
        "text": "BAB I PENDAHULUAN\n\nIni adalah pendahuluan.\n\nA. Latar Belakang\nIni latar belakang.",
        "filename": "tesis.pdf"
    }]
    
    chunks = chunk_documents(pages)
    assert len(chunks) >= 2  # Should split into at least 2 sections


if __name__ == "__main__":
    test_chunker_empty_input()
    test_chunker_single_short_page()
    test_chunker_multiple_pages()
    test_chunker_metadata_preservation()
    test_chunker_empty_page_text()
    test_is_heading_line_bab()
    test_is_heading_letter_sub()
    test_is_heading_numbered()
    test_is_heading_keyword()
    test_is_heading_not()
    test_split_by_headings_simple()
    test_split_by_headings_no_heading()
    test_chunker_with_headings()
    print("✅ All ingestion tests passed!")