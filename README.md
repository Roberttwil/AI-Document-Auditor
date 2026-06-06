# AI Document Auditor 📄🤖

An enterprise-grade, AI-powered document intelligence platform for auditors and financial analysts. Upload complex financial PDFs (prospectuses, annual reports, audit opinions), ask questions in natural language, and receive accurate answers — including computed financial ratios and YoY comparisons — with traceable, clickable source citations.

>  A locally-deployable, LLM-agnostic financial co-pilot that reads borderless tables, performs mathematical analysis, and always shows its sources.

---

## ✨ Key Features

- **High-Fidelity Table Extraction** — Reads borderless financial tables (Income Statement, Balance Sheet, Cash Flow) using vision-AI parsing via LlamaParse
- **Multi-Hop Analytical Reasoning** — Decomposes complex questions into sub-queries, answers each independently, then synthesizes a coherent response
- **Mathematical Calculation** — Computes financial ratios, YoY percentage changes, and deltas directly from extracted numbers (no spreadsheet needed)
- **Traceable Citations** — Every answer includes clickable page references that jump directly to the source page in the PDF viewer
- **LLM-Agnostic** — Switch between Gemini, Llama 3.3 70B, Mixtral, and more at runtime from the UI
- **Multilingual** — Handles bilingual documents (Bahasa Indonesia + English) natively
- **Local-First Embeddings** — The embedding model runs 100% offline; no document data ever leaves your machine
- **Self-Correcting Agent** — Built-in Reviewer node validates answers and retries on failure (up to 2x)

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                        │
│          Next.js Frontend  ───  FastAPI REST Backend       │
└───────────────┬────────────────────────┬───────────────────┘
                │                        │
                ▼                        ▼
   ┌────────────────────┐    ┌─────────────────────────┐
   │  INGESTION PIPELINE│    │   QUERY PIPELINE        │
   │                    │    │   (LangGraph Agent)     │
   │  PDF → LlamaParse  │    │                         │
   │  → Chunker         │    │  Router → Rewriter      │
   │  → Embedder        │    │  → Decomposer           │
   │  → FAISS Index     │    │  → Researcher (×N)      │
   └─────────┬──────────┘    │  → Synthesizer          │
             │               │  → Reviewer             │
             └──────┬────────┘  → Finalizer            │
                    ▼                                  │
           ┌──────────────────┐                        │
           │  FAISS Vector DB │◄───────────────────────┘
           │  (Local Storage) │
           └──────────────────┘
```

---

## 🛠️ Tech Stack

### Backend

| Layer | Technology | Notes |
| :--- | :--- | :--- |
| Runtime | Python 3.11+ | |
| API Framework | FastAPI | Async REST API with background task queue |
| Agent Orchestration | LangGraph + LangChain | Stateful multi-node graph with conditional routing |
| PDF Parser (Primary) | LlamaParse (LlamaCloud) | Vision-AI enterprise parser |
| PDF Parser (Fallback) | PyMuPDF (fitz) | For image-heavy pages with low text confidence |
| Text Splitter | MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter | Two-stage hierarchical splitting |
| Vector Database | FAISS | Fully local — no external database service |
| Embedding Model | `paraphrase-multilingual-MiniLM-L12-v2` | 384-dim, 50+ languages, runs 100% offline |
| Containerization | Docker + Docker Compose | Single-command deploy |

### Frontend

| Layer | Technology | Notes |
| :--- | :--- | :--- |
| Framework | Next.js (React) | App Router architecture |
| Styling | CSS Modules | Vanilla CSS — full design control |
| PDF Viewer | Native `<iframe>` | Deep-links to exact page on citation click |

### LLM Providers (all free-tier compatible)

| Provider | Models Available | Purpose |
| :--- | :--- | :--- |
| Google AI (Gemini) | Gemini 3.1 Flash Lite, 2.5 Flash, 3 Flash | Primary text LLM |
| Groq | Llama 3.3 70B, Mixtral 8x7B | High-speed alternative |
| OpenRouter | Nemotron Nano 12B VL | Vision model for image pages |
| LlamaCloud | LlamaParse | PDF table extraction |

---

## 📋 Prerequisites

- **Python** 3.11+
- **Node.js** 18+
- **Docker & Docker Compose** (for containerized deployment)
- **API Keys** (all have free tiers):
  - [LlamaCloud API Key](https://cloud.llamaindex.ai/) — for LlamaParse PDF parsing
  - [Google AI API Key](https://aistudio.google.com/apikey) — for Gemini LLMs
  - [Groq API Key](https://console.groq.com/) — for Llama 3.3 / Mixtral
  - [OpenRouter API Key](https://openrouter.ai/keys) — for vision model (optional)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <repo-url>
cd AI-Document-Auditor
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 3a. Run with Docker (Recommended)

```bash
docker-compose up --build
```

The app will be available at:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000

### 3b. Run Locally (Development)

**Backend:**
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
python src/main.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## 📖 How to Use

1. **Upload a Document** — Click the upload area and select a PDF financial report
2. **Wait for Processing** — The system extracts text via LlamaParse, chunks it, generates embeddings, and indexes to FAISS
3. **Ask a Question** — Type your question in the chat input (supports Indonesian and English)
4. **Read the Answer** — The AI responds with analysis, calculations, and formatted tables
5. **Verify Sources** — Click citation buttons (e.g., `📄 Pg. 4`) to jump to the exact source page in the PDF viewer

---

## ⚙️ Configuration

All tunable parameters are in `src/config.py`:

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `CHUNK_SIZE` | 1,000 | Parent chunk size (chars) — full context sent to LLM |
| `CHILD_CHUNK_SIZE` | 400 | Child chunk size (chars) — indexed in FAISS for precision search |
| `CHUNK_OVERLAP` | 150 | Overlap between child chunks to prevent table row truncation |
| `TOP_K_RESULTS` | 40 | Number of parent chunks sent to LLM as context |
| `TOP_K_PER_QUERY` | 25 | Child chunks retrieved per query variation |
| `MIN_SIMILARITY_THRESHOLD` | 0.35 | Minimum cosine similarity score to pass retrieval filter |
| `MULTI_QUERY_COUNT` | 3 | Number of query reformulations for broader recall |
| `MAX_RETRIES` | 2 | Maximum self-correction loops by the Reviewer |

---

## 📁 Project Structure

```
AI-Document-Auditor/
├── frontend/                          # Next.js Frontend
│   ├── src/app/
│   │   ├── page.js                    # Main chat + PDF viewer UI
│   │   ├── page.module.css            # Component styles
│   │   ├── globals.css                # Global styles
│   │   └── layout.js                  # App layout & metadata
│   ├── package.json
│   └── .gitignore
│
├── src/                               # Python Backend
│   ├── main.py                        # Application entry point
│   ├── config.py                      # All configuration & model registry
│   │
│   ├── api/
│   │   └── server.py                  # FastAPI REST endpoints
│   │
│   ├── ingestion/                     # Document Processing Pipeline
│   │   ├── pdf_loader.py              # LlamaParse + PyMuPDF extraction
│   │   ├── chunker.py                 # Two-stage chunking + table header injection
│   │   └── table_parser.py            # Advanced table structure detection
│   │
│   ├── vector_store/                  # Vector Memory
│   │   ├── embeddings.py              # SentenceTransformers embedding wrapper
│   │   └── faiss_db.py                # FAISS index with parent-child retrieval
│   │
│   ├── graph/                         # LangGraph Agent Pipeline
│   │   ├── state.py                   # GraphState & data models
│   │   ├── workflow.py                # Graph construction & routing logic
│   │   ├── tools.py                   # Agent tools
│   │   └── nodes/
│   │       ├── router.py              # Intent classifier (text/image/mixed)
│   │       ├── query_rewriter.py      # Multi-query expansion (3 variations)
│   │       ├── query_decomposer.py    # Multi-hop sub-query decomposition
│   │       ├── researcher.py          # RAG search + LLM answer generation
│   │       ├── image_researcher.py    # Vision LLM for chart/image pages
│   │       └── reviewer.py            # Hallucination guard + policy checker
│   │
│   └── ui/                            # Legacy Streamlit UI (deprecated)
│       ├── chat_panel.py
│       └── pdf_viewer.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example                       # Template for environment variables
└── README.md
```

---

## 🔄 LangGraph Workflow

```
START
  ↓
[Router] → classify intent (text / image / mixed)
  ↓
├─ text  → [Query Rewriter] → [Query Decomposer] → [Text Researcher] ───┐
├─ image → [Image Researcher] ──────────────────────────────────────────┤
└─ mixed → both paths in parallel ──────────────────────────────────────┤
                                                                        ↓
                                                            [Synthesizer] (if multi-hop)
                                                                        ↓
                                                                  [Reviewer]
                                                                    ↓
                                                          ┌─────────┴──────────┐
                                                          │ Pass?              │ Fail?
                                                          ↓                    ↓
                                                        END              [Retry] (max 2x)
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## ⚠️ Important Notes

- **API Keys Required** — The system will not function without at least `GOOGLE_API_KEY` and `LLAMA_CLOUD_API_KEY`
- **LlamaParse Free Tier** — Has monthly page limits; heavy usage may require a paid plan
- **FAISS is Local** — All vector data is stored on disk and reloaded on restart; no external database needed
- **No Real-Time Data** — The system answers only from uploaded and indexed documents

---

## 📄 License

This project is for educational and portfolio demonstration purposes.
