"""LangGraph definition: retrieve -> grade -> synthesize with Corrective-RAG loop."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from cortex.agent.deps import AgentDeps
from cortex.agent.nodes import (
    grade_node,
    refuse_node,
    retrieve_rerank_node,
    rewrite_node,
    route_after_grade,
    route_after_retrieve,
    synthesize_node,
)
from cortex.agent.state import KBAgentState


def build_kb_graph(*, deps: AgentDeps, checkpointer=None):
    """Compile the KB agent graph."""
    graph = StateGraph(KBAgentState)

    graph.add_node("retrieve_rerank", retrieve_rerank_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("refuse", refuse_node)

    graph.add_edge(START, "retrieve_rerank")
    graph.add_conditional_edges(
        "retrieve_rerank",
        route_after_retrieve,
        {"grade": "grade", "synthesize": "synthesize"},
    )
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"synthesize": "synthesize", "rewrite": "rewrite", "refuse": "refuse"},
    )
    graph.add_edge("rewrite", "retrieve_rerank")
    graph.add_edge("synthesize", END)
    graph.add_edge("refuse", END)

    return graph.compile(
        checkpointer=checkpointer,
        name="cortex_kb_agent",
    )
