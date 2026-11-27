"""Offline tests for the LangGraph workflow and LangChain chains (fake LLMs, no API key).
Run: python tests/test_graph.py"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tempfile  # noqa: E402

import db  # noqa: E402

db.DB = tempfile.mktemp(suffix=".db")  # tests never touch your real coach.db
db.init()
import interview_graph as ig  # noqa: E402
import llm  # noqa: E402

CFG = {"configurable": {"api_key": "x", "model": "m"}}


class FakeBE:
    def __init__(self, scores, fail_ask_on=None):
        self.scores, self.modes, self.fail_ask_on, self.n_ask = list(scores), [], fail_ask_on, 0

    def retrieve(self, company, role, query="", k=4):
        return [f"ctx:{company}:{query}"]

    def next_question(self, resume, role, company, history, mode="normal",
                      prev_topic="", kb_context=None):
        self.n_ask += 1
        if self.fail_ask_on == self.n_ask:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        self.modes.append(mode)
        return dict(question=f"Q{self.n_ask}", topic=f"T{self.n_ask}", difficulty="easy",
                    focus="F", kind=mode)

    def evaluate(self, q, a, role, company, spoken=False):
        v = self.scores.pop(0)
        return dict(technical_accuracy=v, clarity=v, communication=v, depth=v,
                    weak_points=[], feedback="f", ideal_answer="i")

    def judge(self, q, a, sc):
        return dict(fairness=8, feedback_quality=8, grounded=9, comment="ok")


def run(be, total, judge_on=False):
    ig.BACKENDS["fake"] = lambda: be
    st = ig.GRAPH.invoke(ig.start_state("resume", "DA", "TCS", total, "fake"), CFG)
    return st


def answer(st, text="ans", judge_on=False):
    return ig.GRAPH.invoke({**st, "answer": text, "judge_on": judge_on}, CFG)


def test_adaptive_routing():
    be = FakeBE([3, 3, 9, 6])
    st = run(be, total=4)
    assert st["current_q"]["question"] == "Q1" and st["kb_context"]
    for _ in range(4):
        st = answer(st)
    # weak -> followup; weak again but never 2 deep-dives in a row -> normal;
    # strong -> harder; then finished (no 5th question asked)
    assert be.modes == ["normal", "followup", "normal", "harder"], be.modes
    assert st["done"] and len(st["history"]) == 4 and be.n_ask == 4


def test_judge_optional_and_failsafe():
    be = FakeBE([6, 6])
    st = run(be, total=2)
    st = answer(st, judge_on=True)
    assert st["history"][-1]["judgment"]["grounded"] == 9
    be.judge = lambda *a: (_ for _ in ()).throw(RuntimeError("judge down"))
    st = answer(st, judge_on=True)          # judge failure must not break the turn
    assert st["done"] and "judgment" not in st["history"][-1]


def test_ask_failure_keeps_scorecard_and_retry_works():
    be = FakeBE([6, 6, 6], fail_ask_on=2)
    st = run(be, total=3)
    st = answer(st)                          # evaluate ok, next question fails (quota)
    assert st["scorecard"]["clarity"] == 6 and st["current_q"] is None
    assert len(st["history"]) == 1
    st = ig.GRAPH.invoke(st, CFG)            # retry: answer is None -> plan/retrieve/ask
    assert st["current_q"]["question"] == "Q3" and len(st["history"]) == 1


def test_langchain_chains_with_fake_chat_model():
    """Prompt templates + structured output wiring, using a fake chat model."""
    import lc_chains
    from langchain_core.runnables import RunnableLambda
    seen = []

    class FakeChat:
        def __init__(self, t): self.t = t

        def with_structured_output(self, schema):
            def fn(prompt_value):
                seen.append(prompt_value.to_string())
                if schema is llm.Question:
                    return schema(question="q?", topic="t", difficulty="easy", focus="OOP")
                return schema(technical_accuracy=15, clarity=-3, communication=5, depth=5,
                              weak_points=["x"], feedback="f", ideal_answer="i")
            return RunnableLambda(fn)

    lc_chains._chat = lambda t: FakeChat(t)
    q = lc_chains.next_question("my resume", "DA", "TCS", [], "normal", "", ["ctx1"])
    assert q["focus"] == "OOP" and q["kind"] == "normal"
    assert "FIRST project" in seen[0] and "my resume" in seen[0] and "ctx1" in seen[0]
    s = lc_chains.evaluate("Q", "A {with braces}", "DA", "TCS", spoken=True)
    assert s["technical_accuracy"] == 10 and s["clarity"] == 0   # clamped to 0..10
    assert "auto-transcribed" in seen[1] and "{with braces}" in seen[1]


def test_retriever():
    import lc_chains
    docs = lc_chains.KBRetriever(company="Amazon", role="DA").invoke("leadership")
    assert docs and any("Leadership" in d.page_content for d in docs)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
