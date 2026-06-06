from typing import Literal, List
from langgraph.graph import StateGraph, END
from src.graph.state import GraphState, Citation
from src.graph.nodes.router import router_node
from src.graph.nodes.query_rewriter import generate_query_variations
from src.graph.nodes.query_decomposer import decompose_query, synthesize_answers
from src.graph.nodes.researcher import research_node
from src.graph.nodes.image_researcher import image_researcher_node
from src.graph.nodes.reviewer import review_node
from src.vector_store.faiss_db import FAISSDocumentStore


# Global reference to the vector store (set during initialization)
_vector_store: FAISSDocumentStore = None


def set_vector_store(store: FAISSDocumentStore):
    """Set the global vector store reference for the workflow."""
    global _vector_store
    _vector_store = store


def researcher_wrapper(state: GraphState) -> GraphState:
    """Wrapper to inject vector_store dependency into researcher_node."""
    return research_node(state, _vector_store)


def image_researcher_wrapper(state: GraphState) -> GraphState:
    """Wrapper for image researcher node."""
    return image_researcher_node(state)


def query_rewriter_wrapper(state: GraphState) -> GraphState:
    """
    Wrapper for Query Rewriter node.
    
    Generates multiple query variations for multi-query retrieval.
    The researcher node will use query_variations to search FAISS
    with broader coverage.
    """
    query = state.get("query", "")
    selected_model = state.get("selected_model", None)
    chat_history = state.get("chat_history", [])
    
    variations = generate_query_variations(query, chat_history=chat_history, model_name=selected_model)
    
    return {
        **state,
        "query": variations[0] if variations else query,
        "query_variations": variations
    }


def query_decomposer_wrapper(state: GraphState) -> GraphState:
    """
    Wrapper for Query Decomposer node (P1 — Multi-Hop).
    
    Breaks complex queries into sub-queries. Each sub-query will be
    independently researched via the researcher node, then synthesized.
    """
    query = state.get("query", "")
    selected_model = state.get("selected_model", None)
    
    sub_queries = decompose_query(query, model_name=selected_model)
    needs_synthesis = len(sub_queries) > 1
    
    return {
        **state,
        "sub_queries": sub_queries,
        "sub_answers": [],
        "needs_synthesis": needs_synthesis,
    }


def query_synthesizer_wrapper(state: GraphState) -> GraphState:
    """
    Wrapper for Query Synthesizer node (P1 — Multi-Hop).
    
    Combines answers from multiple sub-queries into one coherent answer.
    Runs after the researcher has answered all sub-queries.
    Only activates when needs_synthesis is True.
    """
    needs_synthesis = state.get("needs_synthesis", False)
    
    if not needs_synthesis:
        # No synthesis needed — pass through
        return state
    
    query = state.get("query", "")
    sub_answers = state.get("sub_answers", [])
    selected_model = state.get("selected_model", None)
    
    if not sub_answers or len(sub_answers) <= 1:
        return state
    
    # Run synthesis
    synthesized = synthesize_answers(query, sub_answers, model_name=selected_model)
    
    # Extract citations from synthesized answer
    from src.graph.nodes.researcher import _parse_citations
    clean_answer, citations = _parse_citations(synthesized)
    
    return {
        **state,
        "text_answer": clean_answer,
        "text_citations": citations,
    }


def route_after_router(state: GraphState) -> Literal["query_rewriter", "image_researcher", "mixed_researchers"]:
    """
    Conditional routing after router node.
    
    Routes to query_rewriter for text/mixed queries (which need multi-query search),
    or directly to image_researcher for pure image queries (no vector search needed).
    """
    intent = state.get("query_intent", "text")
    image_pages = state.get("image_pages", [])
    
    if intent == "image":
        return "image_researcher"
    else:  # text or mixed
        if not image_pages and intent == "mixed":
            return "query_rewriter"
        return "query_rewriter"


def route_after_query_rewriter(state: GraphState) -> Literal["query_decomposer", "image_researcher"]:
    """
    Conditional routing after query rewriter.
    
    Routes to query_decomposer to check if the query needs multi-hop,
    or directly to image_researcher for pure image queries.
    """
    intent = state.get("query_intent", "text")
    if intent == "text":
        return "query_decomposer"
    else:  # mixed
        image_pages = state.get("image_pages", [])
        if not image_pages:
            return "query_decomposer"
        return "query_decomposer"


def route_after_researchers(state: GraphState) -> Literal["reviewer", "finalize"]:
    """
    Conditional routing after researcher(s) complete.
    
    Routes to reviewer for validation.
    """
    # Always go to reviewer if we have answers
    text_answer = state.get("text_answer")
    image_answer = state.get("image_answer")
    
    if text_answer or image_answer:
        return "reviewer"
    
    return "finalize"


def should_continue_review(state: GraphState) -> Literal["text_researcher", "image_researcher", "finalize"]:
    """
    Conditional edge after reviewer.
    
    Determines whether to retry failed researcher or finalize.
    """
    review_result = state.get("review_result")
    review_target = state.get("review_target", "text")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    # If review passed or max retries reached, finalize
    if review_result is None or review_result.get("passed", True):
        return "finalize"
    
    if retry_count >= max_retries:
        return "finalize"
    
    # Retry the failed track
    if review_target == "text":
        return "text_researcher"
    elif review_target == "image":
        return "image_researcher"
    else:  # mixed - retry both
        return "text_researcher"  # Simplified: retry text first


def build_workflow() -> StateGraph:
    """
    Build the LangGraph workflow with dual-agent routing:
    
    START
      ↓
    [Router] → detect intent (text/image/mixed)
      ↓
    ├─ text → [Text Researcher] ──┐
    ├─ image → [Image Researcher] ┤
    └─ mixed → [Text] + [Image] ──┤
               (parallel)          ↓
                            [Reviewer]
                               ↓
                    ┌──────────┬──────────┐
                    │ Pass?    │ Fail?    │
                    ↓          ↓          │
                  END      [Retry]       │
                            (max 2x) ────┘
    
    Returns:
        Compiled StateGraph ready for invocation.
    """
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("query_rewriter", query_rewriter_wrapper)
    workflow.add_node("query_decomposer", query_decomposer_wrapper)
    workflow.add_node("query_synthesizer", query_synthesizer_wrapper)
    workflow.add_node("text_researcher", researcher_wrapper)
    workflow.add_node("image_researcher", image_researcher_wrapper)
    workflow.add_node("reviewer", review_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Route from router:
    #   - image queries → image_researcher directly (no vector search needed)
    #   - text/mixed queries → query_rewriter first
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "query_rewriter": "query_rewriter",
            "image_researcher": "image_researcher",
            "mixed_researchers": "query_rewriter"
        }
    )
    
    # Route from query_rewriter → query_decomposer (checks if multi-hop needed)
    workflow.add_conditional_edges(
        "query_rewriter",
        route_after_query_rewriter,
        {
            "query_decomposer": "query_decomposer",
            "image_researcher": "image_researcher"
        }
    )
    
    # Route from query_decomposer → text_researcher (researcher handles the query)
    workflow.add_edge("query_decomposer", "text_researcher")
    
    # From text researcher → query_synthesizer (if multi-hop) or reviewer
    def after_text_researcher(state):
        needs_synthesis = state.get("needs_synthesis", False)
        intent = state.get("query_intent", "text")
        
        # If multi-hop with sub_queries, pass through to synthesizer
        if needs_synthesis:
            return "query_synthesizer"
        
        # If mixed with image pages, go to image researcher
        if intent == "mixed" and state.get("image_pages"):
            return "image_researcher"
        
        return "reviewer"
    
    workflow.add_conditional_edges(
        "text_researcher",
        after_text_researcher,
        {
            "query_synthesizer": "query_synthesizer",
            "image_researcher": "image_researcher",
            "reviewer": "reviewer"
        }
    )
    
    # From query_synthesizer → reviewer or image researcher
    def after_synthesizer(state):
        intent = state.get("query_intent", "text")
        if intent == "mixed" and state.get("image_pages"):
            return "image_researcher"
        return "reviewer"
    
    workflow.add_conditional_edges(
        "query_synthesizer",
        after_synthesizer,
        {
            "image_researcher": "image_researcher",
            "reviewer": "reviewer"
        }
    )
    
    # From image researcher → reviewer
    workflow.add_edge("image_researcher", "reviewer")
    
    # From reviewer → finalize or retry
    workflow.add_conditional_edges(
        "reviewer",
        should_continue_review,
        {
            "text_researcher": "text_researcher",
            "image_researcher": "image_researcher",
            "finalize": END
        }
    )
    
    # Compile the workflow
    return workflow.compile()


def run_workflow(
    query: str,
    chat_history: list = None,
    selected_model: str = None,
    pdf_pages: dict = None
) -> dict:
    """
    Convenience function to run the full dual-agent workflow with a query.
    
    Args:
        query: User's question
        selected_model: Display name of the selected Gemini model
        pdf_pages: Optional dict of {page_num: PIL.Image} for image analysis
    
    Returns:
        Final state dict with 'final_answer', 'final_citations', 'retry_count'
    """
    if _vector_store is None:
        return {
            "final_answer": "Vector store belum diinisialisasi. Silakan upload dokumen terlebih dahulu.",
            "final_citations": [],
            "retry_count": 0,
            "review_result": {"passed": False, "issues": ["No vector store"]}
        }
    
    # Build and run workflow
    app = build_workflow()
    
    initial_state: GraphState = {
        # Query & Routing
        "query": query,
        "chat_history": chat_history or [],
        "query_intent": "text",  # Will be set by router
        "image_pages": [],
        
        # Multi-Query
        "query_variations": None,
        
        # Query Decomposition (P1)
        "sub_queries": None,
        "sub_answers": [],
        "needs_synthesis": False,
        
        # Text Track
        "context_chunks": [],
        "text_answer": None,
        "text_citations": None,
        
        # Image Track
        "image_answer": None,
        "image_citations": None,
        
        # Validation
        "review_result": None,
        "review_target": "text",
        "retry_count": 0,
        "max_retries": 2,
        
        # Final Output
        "final_answer": None,
        "final_citations": None,
        
        # UI/Config
        "selected_model": selected_model,
        "pdf_pages": pdf_pages or {}
    }
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    # Merge citations from both tracks
    merged_citations = []
    if final_state.get("text_citations"):
        merged_citations.extend(final_state.get("text_citations") or [])
    if final_state.get("image_citations"):
        merged_citations.extend(final_state.get("image_citations") or [])

    # If reviewer did not set final_citations, fallback to merged citations
    final_citations = final_state.get("final_citations")
    if not final_citations:
        final_citations = merged_citations

    return {
        "final_answer": final_state.get("final_answer", ""),
        "final_citations": final_citations,
        "query_intent": final_state.get("query_intent", "text"),
        "retry_count": final_state.get("retry_count", 0),
        "review_result": final_state.get("review_result", {"passed": True, "issues": []}),
        # Backward-compat alias for UI that still expects draft_citations
        "draft_citations": final_citations
    }