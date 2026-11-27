"""Offline tests (no API key needed): python tests/test_offline.py"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import db, kb, sandbox, voice  # noqa: E402


def test_voice():
    r = voice.analyze("um so basically I think like the join is um good", 6)
    assert r["fillers"] == 4 and r["words"] == 11 and r["wpm"] == 110.0
    assert voice.analyze("hello", None)["wpm"] is None


def test_sql():
    ch = sandbox.SQL_CHALLENGES[0]
    assert sandbox.run_sql("select name from employees where salary > 50000", ch)["passed"] == 1
    assert sandbox.run_sql("select name from employees", ch)["passed"] == 0
    assert not sandbox.run_sql("DROP TABLE employees", ch)["ok"]
    assert sandbox.run_sql(sandbox.SQL_CHALLENGES[3]["ref"], sandbox.SQL_CHALLENGES[3])["passed"] == 1


def test_python():
    ch = sandbox.PY_CHALLENGES[2]
    good = "def max_subarray(nums):\n    best=cur=nums[0]\n    for x in nums[1:]:\n        cur=max(x,cur+x); best=max(best,cur)\n    return best\n"
    assert sandbox.run_python(good, ch)["passed"] == 3
    assert sandbox.run_python("def max_subarray(nums):\n    return 0\n", ch)["passed"] == 0
    assert "Time limit" in sandbox.run_python("def max_subarray(nums):\n    while True: pass\n", ch, timeout=1)["error"]
    assert not sandbox.run_python("def max_subarray(:", ch)["ok"]


def test_kb():
    assert "TCS" in kb.companies()
    docs = kb.retrieve("Amazon", "Data Analyst", "leadership")
    assert docs and any("Leadership" in d for d in docs)


def test_db():
    db.DB = tempfile.mktemp(suffix=".db")
    db.init()
    sid = db.new_session("DA", "TCS")
    q = {"topic": "SQL", "difficulty": "easy", "question": "q"}
    s = dict(technical_accuracy=7, clarity=6, communication=6, depth=5)
    aid = db.save_answer(sid, q, "a", s, 6, {"mode": "voice", "wpm": 130, "fillers": 2, "words": 40})
    assert len(db.all_answers()) == 1 and len(db.unjudged()) == 1
    db.save_judgment(aid, dict(fairness=8, feedback_quality=7, grounded=9, comment="ok"))
    assert db.unjudged().empty and len(db.judgments_df()) == 1
    db.log_call("evaluate", 123.4, True)
    assert len(db.calls_df()) == 1
    # visitors must never see each other's data
    sid2 = db.new_session("SDE", "Amazon", owner="visitor2")
    db.save_answer(sid2, q, "secret", s, 6)
    db.save_code_attempt("sql1", "SQL", 1, 1, "visitor2")
    assert len(db.all_answers("local")) == 1 and len(db.all_answers("visitor2")) == 1
    assert db.all_answers("visitor2")["answer"].iloc[0] == "secret"
    assert db.code_df("local").empty and len(db.code_df("visitor2")) == 1
    assert db.all_answers("nobody").empty and db.judgments_df("visitor2").empty


def test_key_isolation():
    import llm
    import threading
    seen = {}

    def worker(name, key):
        llm.set_key(key)
        seen[name] = llm._client()
    ts = [threading.Thread(target=worker, args=(n, k)) for n, k in (("a", "key-A"), ("b", "key-B"))]
    [x.start() for x in ts]
    [x.join() for x in ts]
    assert seen["a"] is not seen["b"]




def test_question_coverage():
    """Questions must rotate project -> fundamentals(OOP) -> project..., and never
    deep-dive twice in a row (mock LLM, no API key)."""
    import llm
    prompts = []

    def fake(kind, prompt, schema, temp):
        prompts.append(prompt)
        return schema(question="q", topic="t", difficulty="easy", focus="f")
    llm._generate = fake
    hist = []
    for _ in range(4):
        q = llm.next_question("resume", "DA", "TCS", hist, "normal")
        hist.append(q)
    assert "FIRST project" in prompts[0]
    assert "OOP" in prompts[1] and "FUNDAMENTALS" in prompts[1]
    assert "DIFFERENT project" in prompts[2]
    assert "ROLE SKILL" in prompts[3]
    # weak answer -> follow-up allowed once, then forced back to a normal question
    q = llm.next_question("resume", "DA", "TCS", hist, "followup", "t")
    assert q["kind"] == "followup"
    hist.append(q)
    q2 = llm.next_question("resume", "DA", "TCS", hist, "harder", "t")
    assert q2["kind"] == "normal"


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
