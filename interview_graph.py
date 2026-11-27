"""LangGraph workflow for ONE interview turn.

First call (no answer yet):  plan -> retrieve -> ask
Later calls (answer given):  evaluate -> [judge] -> (finish | plan -> retrieve -> ask)

The graph is a state machine with conditional routing: the score of the last answer
decides the next move (cross-question, harder question, or move to a new area).
State between turns is kept by the caller (Streamlit session)."""
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import llm

METRICS = ["technical_accuracy", "clarity", "communication", "depth"]


def _lc():
    import lc_chains
    return lc_chains


# backend name -> module exposing next_question / evaluate / judge / review_code / retrieve
BACKENDS = {"langchain": _lc, "sdk": lambda: llm}


class InterviewState(TypedDict, total=False):
    backend: str
    resume: str
    role: str
    company: str
    total: int
    history: list          # answered questions with scores (+ optional judgment)
    current_q: Optional[dict]
    answer: Optional[str]
    spoken: bool
    judge_on: bool
    kb_context: list
    next_mode: str         # normal | followup | harder
    scorecard: Optional[dict]
    done: bool
    ask_error: Optional[str]


def start_state(resume, role, company, total, backend="langchain"):
    return InterviewState(backend=backend, resume=resume, role=role, company=company,
                          total=total, history=[], current_q=None, answer=None,
                          spoken=False, judge_on=False, kb_context=[], next_mode="normal",
                          scorecard=None, done=False, ask_error=None)


def _be(state, config):
    """Bind the caller's API key/model (passed via config, not state) and return the backend."""
    cfg = (config or {}).get("configurable", {})
    llm.set_key(cfg.get("api_key", ""))
    llm.set_model(cfg.get("model", ""))
    return BACKENDS[state.get("backend", "langchain")]()


# ---------------- nodes ----------------
def evaluate_node(state, config):
    be, q = _be(state, config), state["current_q"]
    s = be.evaluate(q["question"], state["answer"], state["role"], state["company"],
                    spoken=state.get("spoken", False))
    avg = sum(s[m] for m in METRICS) / 4
    entry = {**q, **s, "answer": state["answer"], "avg": avg}
    return {"scorecard": s, "history": state.get("history", []) + [entry]}


def judge_node(state, config):
    """Optional audit step; a failure here must never break the interview."""
    try:
        be, entry = _be(state, config), state["history"][-1]
        j = be.judge(entry["question"], entry["answer"], state["scorecard"])
        return {"history": state["history"][:-1] + [{**entry, "judgment": j}]}
    except Exception:
        return {}


def finish_node(state):
    return {"done": True}


def plan_node(state):
    """Adaptive difficulty: weak -> cross-question, strong -> harder, else new area.
    Never two deep-dives in a row."""
    hist = state.get("history", [])
    if not hist:
        return {"next_mode": "normal"}
    last = hist[-1]
    mode = "followup" if last["avg"] < 5 else "harder" if last["avg"] >= 7.5 else "normal"
    if last.get("kind", "normal") != "normal":
        mode = "normal"
    return {"next_mode": mode}


def retrieve_node(state, config):
    be = _be(state, config)
    hist = state.get("history", [])
    query = hist[-1]["topic"] if hist else ""
    return {"kb_context": be.retrieve(state["company"], state["role"], query)}


def ask_node(state, config):
    """If question generation fails (e.g. quota), keep the scorecard and let the caller
    retry later: current_q=None signals 'no next question yet'."""
    try:
        be, hist = _be(state, config), state.get("history", [])
        prev_topic = hist[-1]["topic"] if hist else ""
        q = be.next_question(state["resume"], state["role"], state["company"], hist,
                             state.get("next_mode", "normal"), prev_topic,
                             state.get("kb_context"))
        return {"current_q": q, "answer": None, "ask_error": None}
    except Exception as e:
        return {"current_q": None, "answer": None, "ask_error": str(e)}


# ---------------- routing ----------------
def route_start(state):
    return "evaluate" if state.get("answer") is not None else "plan"


def _finished(state):
    return len(state.get("history", [])) >= state.get("total", 5)


def route_after_evaluate(state):
    if state.get("judge_on"):
        return "judge"
    return "finish" if _finished(state) else "plan"


def route_after_judge(state):
    return "finish" if _finished(state) else "plan"


def build_graph():
    g = StateGraph(InterviewState)
    g.add_node("evaluate", evaluate_node)
    g.add_node("judge", judge_node)
    g.add_node("finish", finish_node)
    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("ask", ask_node)
    g.add_conditional_edges(START, route_start, {"evaluate": "evaluate", "plan": "plan"})
    g.add_conditional_edges("evaluate", route_after_evaluate,
                            {"judge": "judge", "finish": "finish", "plan": "plan"})
    g.add_conditional_edges("judge", route_after_judge, {"finish": "finish", "plan": "plan"})
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "ask")
    g.add_edge("ask", END)
    g.add_edge("finish", END)
    return g.compile()


GRAPH = build_graph()


def mermaid():
    return GRAPH.get_graph().draw_mermaid()
