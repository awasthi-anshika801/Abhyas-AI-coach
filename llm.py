import os
import threading
import time

from google import genai
from google.genai import types
from pydantic import BaseModel

import db

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class Question(BaseModel):
    question: str
    topic: str       # e.g. "SQL Joins", "Behavioral", "Project: YOLOv8"
    difficulty: str  # easy | medium | hard
    focus: str       # project name or subject area, e.g. "Kavach", "OOP", "DBMS", "Behavioral"


class Scorecard(BaseModel):
    technical_accuracy: int  # 0-10
    clarity: int
    communication: int
    depth: int
    weak_points: list[str]
    feedback: str
    ideal_answer: str        # STAR format for behavioral questions


class Judgment(BaseModel):
    fairness: int          # 0-10: are the scores calibrated to the answer's real quality?
    feedback_quality: int  # 0-10: specific, actionable, useful?
    grounded: int          # 0-10: claims supported by question/answer, no hallucination?
    comment: str


class CodeReview(BaseModel):
    time_complexity: str
    space_complexity: str
    feedback: str
    improvement: str


_clients = {}
_local = threading.local()  # per-session key/model, so concurrent visitors never mix


def set_key(key):
    _local.key = (key or "").strip()


def set_model(model):
    _local.model = (model or "").strip()


def _model():
    return getattr(_local, "model", "") or MODEL


def current_model():
    return _model()


def current_key():
    key = getattr(_local, "key", "") or os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Gemini API key missing")
    return key


def _client():
    """Cached client per key: a temporary Client gets garbage-collected and closes its
    HTTP connection ('Cannot send a request, as the client has been closed')."""
    key = getattr(_local, "key", "") or os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Gemini API key missing")
    if key not in _clients:
        # JSON/structured output needs v1beta; some setups default to v1
        _clients[key] = genai.Client(
            api_key=key, http_options=types.HttpOptions(api_version="v1beta"))
    return _clients[key]


TRANSIENT = ("503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED", "500 INTERNAL",
             "overloaded", "high demand")


def is_transient(err):
    """Temporary server-side problems (busy/overloaded model). NOT quota (429) or bad request."""
    msg = str(err)
    return "429" not in msg and "RESOURCE_EXHAUSTED" not in msg and \
        any(t in msg for t in TRANSIENT)


def _timed(kind, fn, retries=2, delay=3.0):
    """Run fn() with automatic retry on transient errors (503 etc.), and log total
    latency + success to SQLite (our lightweight tracing)."""
    t0 = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            out = fn()
            db.log_call(kind, (time.perf_counter() - t0) * 1000, True)
            return out
        except Exception as e:
            if attempt < retries and is_transient(e):
                time.sleep(delay * (attempt + 1))
                continue
            db.log_call(kind, (time.perf_counter() - t0) * 1000, False)
            if is_transient(e):
                raise RuntimeError(
                    "Gemini model abhi busy hai (503). 1-2 minute baad dobara try karo, "
                    "ya sidebar me model badlo (jaise gemini-2.5-flash-lite). "
                    f"Details: {str(e)[:150]}") from e
            raise


def _generate(kind, prompt, schema, temperature):
    def run():
        r = _client().models.generate_content(
            model=_model(), contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema, temperature=temperature))
        return r.parsed
    return _timed(kind, run)


# Coverage plan: normal questions cycle through these slots so the interview
# spreads across ALL projects, core fundamentals, role skills and behavioral.
SLOTS = [
    "PROJECT: ask about the FIRST project listed in the resume. Verify a specific claim: "
    "why a tool/model was chosen, a design trade-off, a challenge, or how they would scale it.",
    "FUNDAMENTALS: ask a core-concepts question. Start with OOP concepts (inheritance, "
    "polymorphism, encapsulation, abstraction) if OOP was not asked yet; otherwise pick another "
    "subject (DBMS/SQL, Python, DSA, statistics, ML basics, OS/networks) fitting the role and resume skills.",
    "PROJECT: ask about a DIFFERENT project than any asked so far (the second project listed). "
    "If the resume has only one project, ask about a different aspect of it.",
    "ROLE SKILL: ask a practical scenario-based question on a key skill for the target role, "
    "using tools/skills listed in the resume.",
    "PROJECT: ask about the THIRD project (or an internship/experience item). If fewer exist, "
    "pick the project asked about the least so far.",
    "FUNDAMENTALS: ask a core-concepts question on a subject DIFFERENT from earlier fundamentals "
    "questions (e.g. DBMS/SQL, Python, DSA, statistics, ML basics).",
    "BEHAVIORAL: ask a behavioral question (teamwork, conflict, failure, leadership) "
    "answerable in STAR format.",
]


# ---- Prompt templates: single source of truth, shared by the raw-SDK backend (this
# file) and the LangChain backend (lc_chains.py, via ChatPromptTemplate). ----
QUESTION_PROMPT = """You are a strict but fair interviewer hiring for: {role}
Company: {company}

Company knowledge base (typical patterns/questions for this company; use as
inspiration and adapt to the role and resume, do not copy blindly):
{ctx}

Candidate resume:
{resume}

Questions asked so far:
{asked}
Focus areas already covered: {covered}

Task for this question: {instr}
Rules: ask exactly ONE question, under 50 words. Never repeat an earlier question. Set
`focus` to the project name or subject area you are asking about (e.g. the exact project
title from the resume, 'OOP', 'DBMS', 'Behavioral')."""

EVALUATE_PROMPT = """You are an expert interviewer for {role} at {company}.
{note}Score the answer 0-10 on each of: technical_accuracy, clarity, communication, depth.
Be honest, do not inflate. Empty or irrelevant answers get 0-2.
List weak_points (short phrases), give 2-3 sentences of feedback, and write an
ideal_answer (STAR format if the question is behavioral).

Question: {question}
Candidate answer: {answer}"""

JUDGE_PROMPT = """You are auditing an AI interview evaluator. Rate it 0-10 on:
- fairness: are the scores calibrated to the answer's real quality (not inflated/harsh)?
- feedback_quality: is the feedback specific and actionable?
- grounded: is every claim supported by the question/answer (no hallucination)?
Add a one-sentence comment.

Question: {question}
Candidate answer: {answer}
Evaluator output: {scorecard}"""

REVIEW_PROMPT = """Review this solution to "{title}" ({passed}/{total} tests passed).
Give its time_complexity and space_complexity (Big-O), short feedback, and one concrete
improvement (an optimal approach if this is not already optimal).

Code:
{code}"""

SPOKEN_NOTE = ("The answer was spoken and auto-transcribed: ignore transcription typos "
               "and filler words when judging content.\n")


def question_vars(resume, role, company, history, mode="normal",
                  prev_topic="", kb_context=None):
    """Template variables for QUESTION_PROMPT. Also returns the effective `mode`.
    At most ONE deep-dive in a row: after a follow-up/harder question the next one
    is a normal question that moves to a new area."""
    if mode != "normal" and history and history[-1].get("kind", "normal") != "normal":
        mode = "normal"
    asked = "\n".join(f"- [{h.get('focus', h['topic'])}] {h['question']}"
                      for h in history) or "none yet"
    covered = ", ".join(dict.fromkeys(h.get("focus", h["topic"]) for h in history)) or "none"
    ctx = "\n".join(f"- {c}" for c in (kb_context or [])) or "none"
    regular = sum(1 for h in history if h.get("kind", "normal") == "normal")
    instr = {
        "normal": SLOTS[regular % len(SLOTS)],
        "followup": f"The candidate answered weakly on '{prev_topic}'. Ask a probing follow-up "
                    "cross-question on the SAME topic and focus, like a real interviewer.",
        "harder": f"The candidate answered strongly on '{prev_topic}'. Ask a harder, deeper "
                  "question on the SAME focus (edge cases, optimization, trade-offs).",
    }[mode]
    return dict(role=role, company=company, ctx=ctx, resume=resume[:8000], asked=asked,
                covered=covered, instr=instr, mode=mode)


def clamp_scorecard(s):
    for k in ("technical_accuracy", "clarity", "communication", "depth"):
        s[k] = max(0, min(10, int(s[k])))
    return s


def clamp_judgment(j):
    for k in ("fairness", "feedback_quality", "grounded"):
        j[k] = max(0, min(10, int(j[k])))
    return j


def next_question(resume, role, company, history, mode="normal",
                  prev_topic="", kb_context=None):
    """mode: normal | followup (weak answer) | harder (strong answer)."""
    v = question_vars(resume, role, company, history, mode, prev_topic, kb_context)
    mode = v.pop("mode")
    out = _generate("question", QUESTION_PROMPT.format(**v), Question, 0.8).model_dump()
    out["kind"] = mode
    return out


def evaluate(question, answer, role, company, spoken=False):
    prompt = EVALUATE_PROMPT.format(role=role, company=company, question=question,
                                    answer=answer, note=SPOKEN_NOTE if spoken else "")
    return clamp_scorecard(_generate("evaluate", prompt, Scorecard, 0.2).model_dump())


def judge(question, answer, scorecard):
    """LLM-as-a-Judge: audits the evaluator's output (meta-evaluation)."""
    prompt = JUDGE_PROMPT.format(question=question, answer=answer, scorecard=scorecard)
    return clamp_judgment(_generate("judge", prompt, Judgment, 0.0).model_dump())


def review_code(title, code, passed, total):
    prompt = REVIEW_PROMPT.format(title=title, code=code, passed=passed, total=total)
    return _generate("code_review", prompt, CodeReview, 0.2).model_dump()


def retrieve(company, role, query="", k=4):
    import kb
    return kb.retrieve(company, role, query, k)


def transcribe(audio_bytes, mime="audio/wav"):
    def run():
        r = _client().models.generate_content(
            model=_model(),
            contents=[types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                      "Transcribe this audio verbatim, INCLUDING filler words such as "
                      "um, uh, like, basically. Return only the transcript."])
        return (r.text or "").strip()
    return _timed("transcribe", run)