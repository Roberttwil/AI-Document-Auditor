from typing import List, Dict, Optional, TypedDict, Literal


class Citation(TypedDict):
    """A single citation with source information."""
    page: int
    filename: str
    snippet: str


class ReviewResult(TypedDict):
    """Result from the reviewer node."""
    passed: bool
    issues: List[str]


class GraphState(TypedDict):
    """
    State object for the LangGraph workflow with dual-agent routing.
    
    Flow:
        START 
          → Router (detect intent: text/image/mixed)
          → Query Rewriter (multi-query expansion)
          → Text Researcher (if text/mixed)
          → Image Researcher (if image/mixed, via OpenRouter)
          → Reviewer (validate both tracks)
          → (retry if needed)
          → END
    """
    # Query & Routing
    query: str                                          # User's question
    chat_history: Optional[List[Dict[str, str]]]       # Previous chat messages from frontend
    query_intent: Literal['text', 'image', 'mixed']    # Router output: what type of query
    image_pages: List[int]                             # Pages mentioned in image queries
    
    # Multi-Query (added by Query Rewriter)
    query_variations: Optional[List[str]]              # Expanded queries for multi-query search
    
    # Query Decomposition (P1 — Multi-Hop)
    sub_queries: Optional[List[str]]                   # Decomposed sub-questions
    sub_answers: Optional[List[Dict]]                  # [{sub_query, answer, citations}, ...]
    needs_synthesis: bool                              # True if sub_queries > 1
    
    # Text Track
    context_chunks: List[Dict]                         # Retrieved chunks from FAISS (text only)
    text_answer: Optional[str]                         # Answer from Text Researcher
    text_citations: Optional[List[Citation]]           # Citations from text
    
    # Image Track
    image_answer: Optional[str]                        # Answer from Image Researcher (OpenRouter)
    image_citations: Optional[List[Citation]]          # Citations from images
    
    # Validation
    review_result: Optional[ReviewResult]              # Reviewer's assessment
    review_target: Literal['text', 'image', 'mixed']  # Which track(s) being reviewed
    retry_count: int                                   # How many times we've retried
    max_retries: int                                   # Maximum retries allowed (default: 2)
    
    # Final Output
    final_answer: Optional[str]                        # Final approved answer (merged if mixed)
    final_citations: Optional[List[Citation]]          # All citations (merged)
    
    # UI/Config
    selected_model: Optional[str]                      # Selected text model (Gemini/Groq)
    pdf_pages: Optional[Dict]                          # {page_num: PIL.Image} for image queries
