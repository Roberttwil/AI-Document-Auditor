import re
from typing import List, Dict
from src.config import CHUNK_SIZE, CHILD_CHUNK_SIZE, CHUNK_OVERLAP
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ─── Splitters ────────────────────────────────────────────────────────
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, 
    strip_headers=False
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)

# ─── Main Chunking Entry Point ────────────────────────────────────────

def chunk_documents(pages: List[Dict]) -> List[Dict]:
    """
    Split page texts into structured chunks based on Markdown hierarchy.
    
    P2 — Parent-Child Retrieval:
    - Produces BOTH parent chunks (Markdown sections, untuk LLM context)
      and child chunks (250 chars, untuk FAISS precision search).
    - Each child chunk stores parent_id + parent_text for expansion.
    - Tables stay as single parent chunks (no child splitting).
    
    Chunking strategy:
    1. By markdown heading using MarkdownHeaderTextSplitter
    2. Table blocks kept intact
    3. Each text parent generates overlapping child chunks
    """
    all_chunks = []
    chunk_idx = 0
    child_count = 0
    parent_count = 0
    global_table_idx = 0

    for page in pages:
        page_text = page["text"].strip()
        filename = page["filename"]
        page_num = page["page_num"]

        if not page_text:
            continue

        # Split page text into parent chunks using Markdown structure
        md_docs = markdown_splitter.split_text(page_text)
        
        for doc in md_docs:
            parent_text = doc.page_content.strip()
            if not parent_text:
                continue

            # Detect if this block is mostly a table (has markdown table syntax)
            # A simple heuristic: has a header separator like "|---|---|"
            is_table = bool(re.search(r'\|.*---.*\|', parent_text))
            
            # Additional check: Vision-extracted tables
            is_vision_table = bool(re.match(r'^\[TABLE_(\d+)\]', parent_text))
            
            if is_table or is_vision_table:
                global_table_idx += 1
                content_type = "table"
            else:
                content_type = "text"
            
            p_id = f"{filename}_{page_num}_{chunk_idx}"
            p_chunk = {
                "chunk_id": p_id,
                "text": parent_text,
                "level": "parent",
                "metadata": {
                    "filename": filename,
                    "page_num": page_num,
                    "chunk_index": chunk_idx,
                    "table_index": global_table_idx if content_type == "table" else None,
                    "content_type": content_type,
                    "level": "parent"
                }
            }
            all_chunks.append(p_chunk)
            parent_count += 1
            
            # Generate child chunks for ALL parents (including tables) so FAISS can find them
            if len(parent_text) >= 300:
                first_line = parent_text.split('\n')[0][:150]
                
                # Metadata Tagging untuk Tabel (ekstrak 2 baris pertama)
                table_header = ""
                if content_type == "table":
                    lines = parent_text.split('\n')
                    table_lines = [l for l in lines if l.strip().startswith('|')]
                    if len(table_lines) >= 2:
                        table_header = table_lines[0] + "\n" + table_lines[1] + "\n"
                        
                child_docs = child_splitter.split_text(parent_text)
                for c_idx, c_text in enumerate(child_docs):
                    if not c_text.strip():
                        continue
                        
                    # Inject filename and context header so isolated rows match semantic queries like "Adaro 2026"
                    if content_type == "table" and table_header and table_header not in c_text:
                        enriched_text = f"[Document: {filename}] [Context: {first_line}]\n{table_header}{c_text}"
                    else:
                        enriched_text = f"[Document: {filename}] [Context: {first_line}]\n{c_text}"
                    
                    c_chunk = {
                        "chunk_id": f"{p_id}_child_{c_idx}",
                        "text": enriched_text,
                        "level": "child",
                        "metadata": {
                            "filename": filename,
                            "page_num": page_num,
                            "chunk_index": chunk_idx + c_idx + 10000,
                            "table_index": None,
                            "content_type": content_type,
                            "level": "child",
                            "parent_id": p_id,
                            "parent_text": parent_text
                        }
                    }
                    all_chunks.append(c_chunk)
                    child_count += 1
                else:
                    # If too short, just create one pseudo-child chunk
                    enriched_text = f"[Document: {filename}]\n{parent_text}"
                    c_chunk = {
                        "chunk_id": f"{p_id}_child_0",
                        "text": enriched_text,
                        "level": "child",
                        "metadata": {
                            "filename": filename,
                            "page_num": page_num,
                            "chunk_index": chunk_idx + 10000,
                            "table_index": None,
                            "content_type": content_type,
                            "level": "child",
                            "parent_id": p_id,
                            "parent_text": parent_text
                        }
                    }
                    all_chunks.append(c_chunk)
                    child_count += 1
            
            chunk_idx += 1

    if child_count > 0:
        print(f"  🧩 P2 Parent-Child: {parent_count} parents → {child_count} children "
              f"(total {len(all_chunks)} chunks)")

    return all_chunks