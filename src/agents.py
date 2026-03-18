from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from src.config import settings
from src.retriever import EnterpriseRetriever
from src.safety import validate_output


# Shared LLM instance used by the planner, rewriter, reasoner, and validator.
# A low temperature keeps responses more deterministic for enterprise QA flows.
llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.openai_api_key,
    temperature=0.1
)


class AgentState(TypedDict):
    # Original user question
    query: str

    # Query variation generated after a failed retrieval/validation cycle
    rewritten_query: str

    # Short planning note to guide downstream retrieval and reasoning
    plan: str

    # Retrieved LangChain document objects
    docs: list

    # Similarity scores returned by the retriever, if available
    retrieval_scores: list

    # Prompt-ready text assembled from retrieved documents
    context: str

    # Initial answer generated from retrieved context
    draft_answer: str

    # Final validated answer returned to the user
    final_answer: str

    # Tracks whether retrieval/validation succeeded, failed, or needs retry
    validation_status: str

    # Stores the reason for success/failure to guide retries
    validation_feedback: str

    # Current retry count
    iteration: int

    # Maximum allowed retry attempts for the workflow
    max_iterations: int


def _format_docs(docs) -> str:
    """
    Convert retrieved documents into a readable prompt block.
    """
    if not docs:
        return "No relevant context found."

    # Include source metadata so the model can ground answers and cite filenames.
    return "\n\n".join(
        [
            f"Source: {d.metadata.get('source', 'Unknown')}\n{d.page_content}"
            for d in docs
        ]
    )


def planner_node(state: AgentState) -> dict:
    """
    Create an initial retrieval plan from the user's question.
    """
    prompt = f"""
You are a planning agent for an enterprise document QA system.

User question:
{state['query']}

Create a short retrieval plan with:
1. Intent
2. Key topics
3. Evidence needed
4. Whether query rewrite may help

Keep the plan concise and practical.
"""
    plan = llm.invoke(prompt).content.strip()
    return {"plan": plan}


def retrieve_node_factory(retriever: EnterpriseRetriever):
    """
    Factory so the retriever instance can be injected into the graph node.
    """
    def retrieve_node(state: AgentState) -> dict:
        # Prefer the rewritten query after a retry; otherwise use the original question.
        active_query = state["rewritten_query"].strip() if state["rewritten_query"] else state["query"]

        try:
            # Use scored retrieval when supported so later nodes can assess quality.
            scored_results = retriever.get_relevant_documents_with_scores(active_query)
            docs = [doc for doc, _ in scored_results]
            scores = [score for _, score in scored_results]
        except Exception:
            # Fall back to plain retrieval if score-based retrieval is unavailable.
            docs = retriever.get_relevant_documents(active_query)
            scores = []

        context = _format_docs(docs)

        return {
            "docs": docs,
            "retrieval_scores": scores,
            "context": context
        }

    return retrieve_node


def assess_context_node(state: AgentState) -> dict:
    """
    Decide if retrieval quality is good enough to proceed to reasoning.

    Uses retrieval scores when available, otherwise falls back to simple checks.
    """
    docs = state.get("docs", [])
    scores = state.get("retrieval_scores", [])

    if not docs:
        return {
            "validation_status": "insufficient",
            "validation_feedback": "No relevant documents retrieved."
        }

    if scores:
        # Lower score is treated as better in this retriever setup.
        best_score = min(scores)
        if best_score <= settings.retrieval_score_threshold:
            return {
                "validation_status": "sufficient",
                "validation_feedback": f"Best retrieval score {best_score:.4f} passed threshold."
            }
        return {
            "validation_status": "insufficient",
            "validation_feedback": f"Best retrieval score {best_score:.4f} did not pass threshold."
        }

    if state["context"].strip().lower() == "no relevant context found.":
        return {
            "validation_status": "insufficient",
            "validation_feedback": "Context is empty."
        }

    return {
        "validation_status": "sufficient",
        "validation_feedback": "Context available without score-based check."
    }


def rewrite_query_node(state: AgentState) -> dict:
    """
    Rewrite the query when retrieval quality is weak.
    """
    prompt = f"""
You are a query rewrite agent for enterprise semantic retrieval.

Original user question:
{state['query']}

Current retrieval plan:
{state['plan']}

Previous query used:
{state['rewritten_query'] or state['query']}

Retrieved context:
{state['context']}

Feedback:
{state['validation_feedback']}

Rewrite the query to improve retrieval quality.
Return only the rewritten query.
"""
    rewritten_query = llm.invoke(prompt).content.strip()

    # If rewriting fails or returns empty output, keep the original question.
    if not rewritten_query:
        rewritten_query = state["query"]

    return {
        "rewritten_query": rewritten_query,
        "iteration": state["iteration"] + 1
    }


def reason_node(state: AgentState) -> dict:
    """
    Generate a grounded draft answer from retrieved context.
    """
    prompt = f"""
You are a reasoning agent in an enterprise RAG system.

User question:
{state['query']}

Retrieval plan:
{state['plan']}

Retrieved context:
{state['context']}

Instructions:
- Answer only from the retrieved context.
- Be concise and accurate.
- If context is insufficient, say so clearly.
- Include source names where possible.
- Do not invent information.
"""
    draft_answer = llm.invoke(prompt).content.strip()
    return {"draft_answer": draft_answer}


def validator_node(state: AgentState) -> dict:
    """
    Validate grounding and decide whether to approve or retry.
    """
    prompt = f"""
You are a validation agent.

Question:
{state['query']}

Draft answer:
{state['draft_answer']}

Retrieved context:
{state['context']}

Evaluate the answer for:
1. Grounding in retrieved context
2. Hallucination risk
3. Source attribution
4. Whether another retrieval attempt is needed

Return exactly in this format:
STATUS: approved OR retry
FEEDBACK: <brief reason>
ANSWER: <revised final answer>
"""
    result = llm.invoke(prompt).content.strip()

    # Default to retry unless the validator explicitly approves the answer.
    status = "retry"
    feedback = "Validator requested retry."
    answer_lines = []
    answer_started = False

    # Parse the validator's structured response into status, feedback, and answer.
    for line in result.splitlines():
        stripped = line.strip()

        if stripped.startswith("STATUS:"):
            status = stripped.replace("STATUS:", "").strip().lower()
            answer_started = False
        elif stripped.startswith("FEEDBACK:"):
            feedback = stripped.replace("FEEDBACK:", "").strip()
            answer_started = False
        elif stripped.startswith("ANSWER:"):
            answer_lines.append(stripped.replace("ANSWER:", "").strip())
            answer_started = True
        elif answer_started:
            answer_lines.append(stripped)

    candidate_answer = "\n".join(answer_lines).strip()
    if not candidate_answer:
        candidate_answer = state["draft_answer"]

    # Apply application-level safety and grounding checks before approval.
    safe_answer = validate_output(candidate_answer)

    if safe_answer.startswith("Validation failed:"):
        return {
            "validation_status": "retry",
            "validation_feedback": safe_answer,
            "final_answer": ""
        }

    if status == "approved":
        return {
            "validation_status": "approved",
            "validation_feedback": feedback,
            "final_answer": safe_answer
        }

    return {
        "validation_status": "retry",
        "validation_feedback": feedback,
        "final_answer": ""
    }


def fallback_node(state: AgentState) -> dict:
    """
    Final fallback when the graph cannot approve a grounded answer.
    """
    if state.get("draft_answer"):
        # Expose the best available draft so the user still gets a useful hint.
        return {
            "final_answer": (
                "I could not fully validate the answer against the retrieved context. "
                "Please refine the question or upload more relevant documents.\n\n"
                f"Draft answer:\n{state['draft_answer']}"
            )
        }

    return {
        "final_answer": (
            "Unable to generate a sufficiently grounded answer from the available documents."
        )
    }


def route_after_assess(state: AgentState) -> Literal["reason", "rewrite_query", "fallback"]:
    """
    Route after assessing retrieval quality.
    """
    if state["validation_status"] == "sufficient":
        return "reason"

    if state["iteration"] >= state["max_iterations"]:
        return "fallback"

    return "rewrite_query"


def route_after_validate(state: AgentState) -> Literal["end", "rewrite_query", "fallback"]:
    """
    Route after validation.
    """
    if state["validation_status"] == "approved":
        return "end"

    if state["iteration"] >= state["max_iterations"]:
        return "fallback"

    return "rewrite_query"


def build_agent_graph(retriever: EnterpriseRetriever):
    """
    Build and compile the LangGraph autonomous workflow.
    """
    graph = StateGraph(AgentState)

    # Each node represents one stage in the agentic RAG workflow.
    graph.add_node("planner", planner_node)
    graph.add_node("retrieve", retrieve_node_factory(retriever))
    graph.add_node("assess_context", assess_context_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("reason", reason_node)
    graph.add_node("validate", validator_node)
    graph.add_node("fallback", fallback_node)

    # Initial path: understand the question, retrieve evidence, then assess quality.
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "assess_context")

    # If retrieval is weak, try rewriting the query before reasoning.
    graph.add_conditional_edges(
        "assess_context",
        route_after_assess,
        {
            "reason": "reason",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback"
        }
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("reason", "validate")

    # Validation either approves the answer, retries with a better query, or falls back.
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "end": END,
            "rewrite_query": "rewrite_query",
            "fallback": "fallback"
        }
    )

    graph.add_edge("fallback", END)

    return graph.compile()


def run_agentic_pipeline(query: str, retriever: EnterpriseRetriever) -> str:
    """
    Execute the LangGraph-based autonomous agent workflow.
    """
    app = build_agent_graph(retriever)

    # Initialize all fields expected by the graph state.
    initial_state: AgentState = {
        "query": query,
        "rewritten_query": "",
        "plan": "",
        "docs": [],
        "retrieval_scores": [],
        "context": "",
        "draft_answer": "",
        "final_answer": "",
        "validation_status": "",
        "validation_feedback": "",
        "iteration": 0,
        "max_iterations": settings.max_agent_iterations
    }

    result = app.invoke(initial_state)

    # Prefer the fully validated answer when available.
    if result.get("final_answer"):
        return result["final_answer"]

    # If validation could not finalize, return the best draft available.
    if result.get("draft_answer"):
        return result["draft_answer"]

    return "Unable to generate a sufficiently grounded answer from the available documents."
