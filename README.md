# AI Document Auditor 📄🤖

Sistem **Multi-Document Agentic RAG** untuk menganalisis laporan keuangan PDF.  
Upload beberapa dokumen PDF, ajukan pertanyaan, dan AI akan menjawab dengan kutipan presisi + nomor halaman.

## ✨ Fitur Utama

- **Multi-PDF Upload** — Upload beberapa file PDF laporan keuangan sekaligus
- **Agentic RAG dengan Self-Correction** — Researcher → Reviewer loop untuk mengurangi halusinasi
- **Citation dengan Page Jumping** — Setiap jawaban disertai tombol yang langsung melompat ke halaman PDF bukti
- **In-Memory Vector Store** — FAISS di RAM, tanpa database permanen
- **Bahasa Indonesia** — Seluruh antarmuka dan jawaban dalam Bahasa Indonesia

## 🏗️ Arsitektur

```
User Query → LangGraph Workflow
               ├── Researcher Node: Cari konteks di FAISS + Groq LLM → Draft Jawaban
               └── Reviewer Node: Validasi angka & halaman → jika gagal, loop ke Researcher (max 2x retry)
                    └── Output: Jawaban Final + Citation Buttons
```

## 🛠️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Frontend | Streamlit (side-by-side layout) |
| LLM | Groq API (Mixtral 8x7b) |
| Orchestration | LangGraph (State Machine) |
| Embedding | OpenAI text-embedding-3-small |
| Vector Store | FAISS (In-Memory) |
| Deployment | Docker & Docker Compose |

## 📋 Prasyarat

- Python 3.12+
- API Keys:
  - [Groq API Key](https://console.groq.com/) — untuk LLM
  - [OpenAI API Key](https://platform.openai.com/api-keys) — untuk embedding

## 🚀 Instalasi & Menjalankan

### Local Development

1. Clone repositori:
```bash
git clone <repo-url>
cd BEI
```

2. Buat virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependensi:
```bash
pip install -r requirements.txt
```

4. Setup environment variables:
```bash
cp .env.example .env
# Edit .env dengan API keys Anda
```

5. Jalankan aplikasi:
```bash
streamlit run src/main.py
```

Akses di browser: http://localhost:8501

### Docker Deployment

1. Setup environment variables:
```bash
cp .env.example .env
# Edit .env dengan API keys Anda
```

2. Build & jalankan dengan Docker Compose:
```bash
docker-compose up --build
```

Akses di browser: http://localhost:8501

## 📖 Cara Penggunaan

1. **Upload Dokumen** — Klik sidebar kiri, upload file PDF laporan keuangan
2. **Tunggu Proses** — Sistem akan mengekstrak teks, memecah (chunking), dan menyimpan ke vector store
3. **Ajukan Pertanyaan** — Ketik pertanyaan di kolom chat
4. **Lihat Jawaban** — AI akan menjawab dengan kutipan dan nomor halaman
5. **Klik Citation** — Klik tombol "Hal.X" untuk melompat ke halaman PDF yang menjadi bukti

## 🔄 Workflow Diagram

```
START → [Researcher] → [Reviewer] → END (jika lulus review)
                        ↓ (jika gagal & retry < 2)
                     [Researcher] → [Reviewer] → END (paksa output jika retry >= 2)
```

## ⚙️ Konfigurasi

Semua konfigurasi ada di `src/config.py`:

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `CHUNK_SIZE` | 1000 | Ukuran chunk teks (karakter) |
| `CHUNK_OVERLAP` | 200 | Overlap antar chunk |
| `TOP_K_RESULTS` | 5 | Jumlah chunk relevan yang diambil |
| `MAX_RETRIES` | 2 | Maksimal loop self-correction |
| `LLM_MODEL` | mixtral-8x7b-32768 | Model Groq yang digunakan |

## 📁 Struktur Proyek

```
BEI/
├── src/
│   ├── main.py                    # Entry point Streamlit
│   ├── config.py                  # Konfigurasi
│   ├── ingestion/
│   │   ├── pdf_loader.py          # Ekstraksi teks PDF
│   │   └── chunker.py             # Pemecahan teks
│   ├── vector_store/
│   │   ├── embeddings.py          # OpenAI embedding
│   │   └── faiss_db.py            # FAISS in-memory
│   ├── graph/
│   │   ├── state.py               # State definition
│   │   ├── nodes/
│   │   │   ├── researcher.py      # Node pencarian & draft
│   │   │   └── reviewer.py        # Node self-correction
│   │   └── workflow.py            # LangGraph workflow
│   └── ui/
│       ├── chat_panel.py          # Chat interface
│       └── pdf_viewer.py          # PDF viewer
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## 🧪 Testing

```bash
# Jalankan semua test
pytest tests/ -v
```

## ⚠️ Catatan Penting

- **API Keys Required**: Sistem tidak akan berjalan tanpa GROQ_API_KEY dan OPENAI_API_KEY
- **In-Memory Storage**: Semua data hilang saat server restart
- **PDF Max Size**: Upload dibatasi 50MB (konfigurasi di docker-compose.yml)
- **Bahasa**: Sistem dioptimalkan untuk Bahasa Indonesia