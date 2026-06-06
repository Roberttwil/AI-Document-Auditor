from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import uuid
import sys
from typing import List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import get_model_display_names, get_default_model
from src.ingestion.pdf_loader import extract_pdf_text
from src.ingestion.chunker import chunk_documents
from src.vector_store.faiss_db import FAISSDocumentStore
from src.graph.workflow import set_vector_store, run_workflow

app = FastAPI(title="AI Document Auditor API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
vector_store = FAISSDocumentStore()
set_vector_store(vector_store)
TEMP_PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "temp_pdfs")
os.makedirs(TEMP_PDF_DIR, exist_ok=True)


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = None
    model_name: Optional[str] = None


# Global tracking for background upload tasks
upload_tasks = {}

def process_pdf_background(task_id: str, file_path: str, filename: str):
    try:
        upload_tasks[task_id] = {"status": "processing", "filename": filename, "progress": "Membaca file PDF..."}
        
        # Read bytes for extraction
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            
        upload_tasks[task_id]["progress"] = "Mengekstrak teks & tabel via Gemini Vision..."
        from src.config import ENABLE_VISION_INGESTION
        pages = extract_pdf_text(
            file_bytes=file_bytes, 
            filename=filename, 
            enable_table_vision=ENABLE_VISION_INGESTION
        )
        
        upload_tasks[task_id]["progress"] = "Memecah dokumen menjadi potongan kecil (chunking)..."
        chunks = chunk_documents(pages)
        
        upload_tasks[task_id]["progress"] = "Membersihkan memori lama & menyimpan dokumen baru..."
        vector_store.clear()
        count = vector_store.add_documents(chunks)
        
        upload_tasks[task_id] = {
            "status": "completed", 
            "filename": filename, 
            "chunks_added": count,
            "progress": "Selesai"
        }
    except Exception as e:
        upload_tasks[task_id] = {
            "status": "failed", 
            "filename": filename, 
            "error": str(e),
            "progress": f"Gagal: {e}"
        }

@app.post("/api/upload")
async def upload_pdf(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    task_ids = []
    
    for file in files:
        if not file.filename.endswith(".pdf"):
            continue
            
        file_path = os.path.join(TEMP_PDF_DIR, file.filename)
        
        # Save file to temp directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        task_id = str(uuid.uuid4())
        upload_tasks[task_id] = {"status": "queued", "filename": file.filename, "progress": "Menunggu giliran..."}
        
        background_tasks.add_task(process_pdf_background, task_id, file_path, file.filename)
        task_ids.append(task_id)
            
    return JSONResponse(content={
        "task_ids": task_ids, 
        "message": "Upload diproses di background",
        "total_documents": vector_store.total_documents
    })

@app.get("/api/upload/status")
async def get_upload_status():
    """Return the status of all background upload tasks."""
    return JSONResponse(content={
        "tasks": upload_tasks,
        "total_documents": vector_store.total_documents
    })

@app.get("/api/documents")
async def get_documents():
    """Return the list of unique filenames currently in FAISS."""
    filenames = set()
    for meta in vector_store.metadata_store:
        if "filename" in meta:
            filenames.add(meta["filename"])
    return JSONResponse(content={"documents": list(filenames)})

@app.post("/api/chat")
async def chat(request: ChatRequest):
    model = request.model_name or get_default_model()
    
    if vector_store.total_documents == 0:
        return JSONResponse(content={
            "final_answer": "Vector store belum diinisialisasi. Silakan upload dokumen terlebih dahulu.",
            "final_citations": [],
            "retry_count": 0
        })
        
    try:
        # Run langgraph workflow
        # Run workflow
        result = run_workflow(
            query=request.query,
            chat_history=[msg.dict() for msg in request.history] if request.history else [],
            selected_model=model,
            pdf_pages={}
        )
        
        return JSONResponse(content={
            "final_answer": result.get("final_answer"),
            "final_citations": result.get("final_citations", []),
            "retry_count": result.get("retry_count", 0),
            "review_result": result.get("review_result", {})
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pdf/{filename}")
async def get_pdf(filename: str):
    file_path = os.path.join(TEMP_PDF_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(file_path, media_type="application/pdf")


@app.get("/api/models")
async def get_models():
    models = get_model_display_names()
    return JSONResponse(content={"models": models, "default": get_default_model()})


@app.post("/api/reset")
async def reset_store():
    vector_store.clear()
    
    # Delete temp pdfs
    for filename in os.listdir(TEMP_PDF_DIR):
        file_path = os.path.join(TEMP_PDF_DIR, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
            
    return JSONResponse(content={"status": "success", "message": "Store cleared"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
