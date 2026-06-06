import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

# Embedding Settings (Lokal - 100% offline, bebas limit)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384

# Chunking Settings
CHUNK_SIZE = 1000           # Parent chunk size (dikirim ke LLM)
CHILD_CHUNK_SIZE = 400      # Child chunk size (diperbesar agar baris terakhir tabel tidak terpotong)
CHUNK_OVERLAP = 150         # Overlap antar child chunks dalam parent yang sama (diperbesar untuk transisi aman)

# Retrieval Settings
TOP_K_RESULTS = 40          # Jumlah parent chunks final yang dikembalikan
TOP_K_PER_QUERY = 25         # Per sub-query dalam Multi-Query
TOP_K_CHILD = 150            # Jumlah child chunks yang dicari sebelum parent expansion
MAX_RETRIES = 2
MULTI_QUERY_COUNT = 3       # Jumlah reformulasi query (termasuk query asli)
MIN_SIMILARITY_THRESHOLD = 0.35 # Batas minimum Cosine Similarity untuk hasil pencarian (diturunkan agar potongan tabel pendek bisa lolos)

# PDF Ingestion Settings
MIN_TEXT_LENGTH_FOR_TEXT_PAGE = 50
VISION_DESCRIPTION_PROMPT = "Jelaskan secara detail apa yang ada di gambar ini dalam Bahasa Indonesia. Fokus pada diagram, angka, dan teks yang terlihat."
RENDER_DPI = 150
OCR_NOISE_THRESHOLD = 0.5  # Confidence threshold untuk deteksi OCR noise (0-1)

# Jika True, ingestion akan mencoba mendeskripsikan halaman bergambar (teks sangat sedikit) menggunakan Gemini.
# Default False karena sering boros kuota dan kita sudah punya image-agent saat Q&A.
ENABLE_VISION_INGESTION = False

# OpenRouter Settings (untuk image analysis)
OPENROUTER_IMAGE_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
OPENROUTER_MAX_TOKENS = 2048

# Streamlit Settings
APP_TITLE = "AI Document Auditor"
APP_ICON = "📄"

# ─── Available Models ──────────────────────────────────────────────
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    # ── Google Gemini Models ──
    "⚡ Gemini 3.1 Flash Lite (Default)": {
        "provider": "gemini",
        "id": "gemini-3.1-flash-lite",
        "description": "Paling aman dari limit (15 RPM, 500 RPD), sangat cepat",
        "temperature": 0.1,
        "max_tokens": 4096
    },
    "⚡ Gemini 2.5 Flash Lite": {
        "provider": "gemini",
        "id": "gemini-2.5-flash-lite",
        "description": "Ringan dan cepat (10 RPM, 20 RPD)",
        "temperature": 0.1,
        "max_tokens": 2048
    },
    "🔥 Gemini 3 Flash": {
        "provider": "gemini",
        "id": "gemini-3-flash",
        "description": "Performa reasoning lebih baik, limit ketat (5 RPM)",
        "temperature": 0.1,
        "max_tokens": 4096
    },
    "🧠 Gemini 2.5 Flash": {
        "provider": "gemini",
        "id": "gemini-2.5-flash",
        "description": "Versi stabil 2.5, limit sangat ketat (5 RPM)",
        "temperature": 0.1,
        "max_tokens": 8192
    },
    # ── Groq Models (Open-Source LLMs, gratis & cepat) ──
    "🦙 Llama 3.3 70B (Groq)": {
        "provider": "groq",
        "id": "llama-3.3-70b-versatile",
        "description": "Gratis, 128K context, reasoning sangat kuat — setara GPT-4 untuk analisis finansial",
        "temperature": 0.1,
        "max_tokens": 8192
    },
    "🧠 GPT OSS 120B (Groq)": {
        "provider": "groq",
        "id": "openai/gpt-oss-120b",
        "description": "Custom Groq Model",
        "temperature": 0.1,
        "max_tokens": 4096
    },
    "🧩 Mixtral 8x7B (Groq)": {
        "provider": "groq",
        "id": "mixtral-8x7b-32768",
        "description": "Gratis, 32K context, cepat — alternatif Llama yang lebih ringan",
        "temperature": 0.1,
        "max_tokens": 4096
    },
    "🔬 Llama 3.1 8B (Groq)": {
        "provider": "groq",
        "id": "llama-3.1-8b-instant",
        "description": "Gratis, sangat cepat — cocok untuk query rewrite & review",
        "temperature": 0.1,
        "max_tokens": 4096
    },
    # ── OpenRouter Models ──
    "🦙 Llama 3.3 70B Instruct (OpenRouter)": {
        "provider": "openrouter",
        "id": "meta-llama/llama-3.3-70b-instruct:free",
        "description": "Gratis, kuat, API OpenRouter",
        "temperature": 0.1,
        "max_tokens": 8192
    },
    "🧠 GPT OSS 120B (OpenRouter)": {
        "provider": "openrouter",
        "id": "openai/gpt-oss-120b:free",
        "description": "Gratis, API OpenRouter",
        "temperature": 0.1,
        "max_tokens": 8192
    },
    "🌐 Qwen3 Next 80B (OpenRouter)": {
        "provider": "openrouter",
        "id": "qwen/qwen3-next-80b-a3b-instruct:free",
        "description": "Gratis, tangguh untuk logika, API OpenRouter",
        "temperature": 0.1,
        "max_tokens": 8192
    },
}


def _get_model_config(model_name: str) -> tuple:
    """Look up a model config by display name.

    Returns:
        (model_config_dict, provider: str) or raises KeyError.
    """
    if model_name is None or model_name not in AVAILABLE_MODELS:
        default_key = list(AVAILABLE_MODELS.keys())[0]
        return AVAILABLE_MODELS[default_key], default_key
    return AVAILABLE_MODELS[model_name], model_name


def get_llm(model_name: str = None, temperature: float = None, tier: str = "heavy"):
    """
    Factory function to create an LLM instance.

    Automatically routes to the correct fallback hierarchy based on tier.

    Args:
        model_name: Display name from AVAILABLE_MODELS (Ignored, hardcoded by user)
        temperature: Override temperature (default: from model config)
        tier: "heavy" (Tier 1/2) or "light" (Tier 3)

    Returns:
        LangChain LLM instance with fallbacks
    """
    if temperature is None:
        temperature = 0.1
    max_tokens = 8192

    if tier == "light":
        # Tier 3 Hierarchy (Fast & Light) requested by user
        # 1. Qwen3 32B via Groq
        llm1 = _get_groq_llm("qwen/qwen3-32b", temperature, max_tokens)
        # 2. Gemma 4 31B via OpenRouter
        llm2 = _get_openrouter_llm("google/gemma-4-31b-it:free", temperature, max_tokens)
        # 3. Gemma 4 26B via OpenRouter
        llm3 = _get_openrouter_llm("google/gemma-4-26b-a4b-it:free", temperature, max_tokens)
        # 4. Laguna via OpenRouter
        llm4 = _get_openrouter_llm("poolside/laguna-m.1:free", temperature, max_tokens)
        # 5. Gemini Flash Lite via Gemini API (Ultimate Fallback)
        llm5 = _get_gemini_llm("gemini-3.1-flash-lite", temperature, max_tokens)
        return llm1.with_fallbacks([llm2, llm3, llm4, llm5])
    else:
        # Hardcoded hierarchy as requested by user (Tier 1 & 2)
        # 1. GPT OSS via Groq
        llm1 = _get_groq_llm("openai/gpt-oss-120b", temperature, max_tokens)
        # 2. GPT OSS via OpenRouter
        llm2 = _get_openrouter_llm("openai/gpt-oss-120b:free", temperature, max_tokens)
        # 3. Llama via Groq
        llm3 = _get_groq_llm("llama-3.3-70b-versatile", temperature, max_tokens)
        # 4. Llama via OpenRouter
        llm4 = _get_openrouter_llm("meta-llama/llama-3.3-70b-instruct:free", temperature, max_tokens)
        # 5. Qwen via OpenRouter
        llm5 = _get_openrouter_llm("qwen/qwen3-next-80b-a3b-instruct:free", temperature, max_tokens)
        return llm1.with_fallbacks([llm2, llm3, llm4, llm5])


def _get_gemini_llm(model_id: str, temperature: float, max_tokens: int):
    """Create a Google Gemini LLM instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set. Please provide it in .env file or environment variables.")

    return ChatGoogleGenerativeAI(
        model=model_id,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        convert_system_message_to_human=True,
        max_retries=0
    )


def _get_groq_llm(model_id: str, temperature: float, max_tokens: int):
    """Create a Groq LLM instance (free, fast, 128K context)."""
    from langchain_groq import ChatGroq

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Please provide it in .env file or environment variables. "
            "Get yours from: https://console.groq.com/keys"
        )

    return ChatGroq(
        model=model_id,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens
    )


def _get_openrouter_llm(model_id: str, temperature: float, max_tokens: int):
    """Create an OpenRouter LLM instance."""
    from langchain_openrouter import ChatOpenRouter

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Please provide it in .env file or environment variables. "
            "Get it from: https://openrouter.ai/keys"
        )

    return ChatOpenRouter(
        model=model_id,
        openrouter_api_key=OPENROUTER_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens
    )


def get_model_display_names() -> list:
    return list(AVAILABLE_MODELS.keys())


def get_default_model() -> str:
    return list(AVAILABLE_MODELS.keys())[0]


def is_groq_model(model_name: str) -> bool:
    """Check if a model is a Groq model."""
    config, _ = _get_model_config(model_name)
    return config.get("provider") == "groq"


def get_llm_vision(model_name: str = None, temperature: float = 0.1):
    """
    Factory function to create an OpenRouter LLM instance for vision/image analysis.

    Args:
        model_name: Model identifier (default: claude-3.5-sonnet via openrouter)
        temperature: Temperature for generation (default: 0.1 for consistency)

    Returns:
        LLM instance from LangChain
    """
    from langchain_openrouter import ChatOpenRouter

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Please provide it in .env file or environment variables. "
            "Get it from: https://openrouter.ai/keys"
        )

    model = model_name or OPENROUTER_IMAGE_MODEL

    return ChatOpenRouter(
        model=model,
        openrouter_api_key=OPENROUTER_API_KEY,
        temperature=temperature,
        max_tokens=OPENROUTER_MAX_TOKENS
    )


def get_llm_vision_gemini(model_name: str = None, temperature: float = 0.1, max_tokens: int = 2048):
    """Vision-capable Gemini via LangChain.

    Uses the same GOOGLE_API_KEY. Model default is gemini-3.1-flash-lite.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set. Please provide it in .env file or environment variables.")

    model_id = model_name or "gemini-3.5-flash"
    primary_llm = ChatGoogleGenerativeAI(
        model=model_id,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        convert_system_message_to_human=True,
        max_retries=0, # Wajib 0 agar langsung gagal saat limit dan masuk ke fallback
    )
    
    # Automatic fallback for strict rate limits on smart models
    if model_id == "gemini-3.5-flash":
        fallback_models = ["gemini-3-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.1-flash-lite"]
        fallbacks = [
            ChatGoogleGenerativeAI(
                model=m,
                google_api_key=GOOGLE_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
                convert_system_message_to_human=True,
                max_retries=0, # Wajib 0
            ) for m in fallback_models
        ]
        return primary_llm.with_fallbacks(fallbacks)
        
    return primary_llm