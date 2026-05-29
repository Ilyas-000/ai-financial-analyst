"""DocsResearcher subgraph: hybrid retrieve → cross-encoder rerank → summarize.

Linear path. ``rerank`` short-circuits to RRF pass-through when
``enable_rerank`` is false or retrieval returned nothing. ``summarize`` calls
the specialist LLM with ``prompts/docs_researcher.txt``.

Contract (``DocsResearcherState``):
    in:  ``question``, ``user_role``, ``company_id``
    out: ``summary`` (Russian), ``sources`` (numbered refs), ``top_chunks``
"""

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.graph.llm import FINAL_ANSWER_TAG, invoke_llm, make_llm
from app.graph.prompts import load_two_section_prompt
from app.rag.reranker import rerank_async
from app.rag.retriever import RetrievedChunk, hybrid_search
from app.rag.tenants import tenant_slug_for

_PROMPT_FILE = "docs_researcher.txt"

_NO_CONTEXT_FALLBACK = (
    "В доступных регламентах ответа на этот вопрос нет. "
    "Уточните формулировку либо запросите structured-проверку по фактическим данным."
)


class DocsResearcherState(TypedDict, total=False):
    question: str
    user_role: str
    company_id: int
    tenant_slug: str
    candidates: list[RetrievedChunk]
    top_chunks: list[RetrievedChunk]
    summary: str
    sources: list[dict]


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        section = f" — {chunk.section}" if chunk.section else ""
        parts.append(f"[{index}] {chunk.doc_title}{section}\n{chunk.text}")
    return "\n\n".join(parts)


def _format_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "n": index,
            "doc_id": chunk.doc_id,
            "doc_title": chunk.doc_title,
            "section": chunk.section,
            "tenant_scope": chunk.tenant_scope,
            "score": chunk.score,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


async def _retrieve_node(state: DocsResearcherState) -> DocsResearcherState:
    slug = state.get("tenant_slug") or await tenant_slug_for(state["company_id"])
    chunks = await hybrid_search(
        question=state["question"],
        user_role=state["user_role"],
        tenant_slug=slug,
    )
    return {"tenant_slug": slug, "candidates": chunks}


async def _rerank_node(state: DocsResearcherState) -> DocsResearcherState:
    settings = get_settings()
    candidates = state.get("candidates", [])
    if not settings.enable_rerank or not candidates:
        return {"top_chunks": list(candidates[: settings.rerank_output_top_k])}
    top = await rerank_async(
        state["question"], candidates, settings.rerank_output_top_k
    )
    return {"top_chunks": top}


async def _summarize_node(state: DocsResearcherState) -> DocsResearcherState:
    chunks = state.get("top_chunks", [])
    if not chunks:
        return {"summary": _NO_CONTEXT_FALLBACK, "sources": []}
    system_prompt, user_template = load_two_section_prompt(_PROMPT_FILE)
    user_prompt = user_template.format(
        question=state["question"], context=_format_context(chunks)
    )
    llm = make_llm("specialist", tags=[FINAL_ANSWER_TAG])
    response = await invoke_llm(llm, [SystemMessage(system_prompt), HumanMessage(user_prompt)])
    return {
        "summary": str(response.content).strip(),
        "sources": _format_sources(chunks),
    }


def build_docs_researcher_subgraph():
    graph = StateGraph(DocsResearcherState)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("rerank", _rerank_node)
    graph.add_node("summarize", _summarize_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()
