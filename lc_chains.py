"""LangChain backend: the same LLM steps as llm.py, built as LCEL chains
(ChatPromptTemplate | ChatGoogleGenerativeAI.with_structured_output(PydanticModel)),
plus a LangChain Retriever over the company knowledge base."""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

import kb
import llm

_models = {}


def _chat(temperature):
    """Cached chat model per (key, model, temperature). max_retries=1 so a daily-quota
    429 fails fast instead of waiting through long backoff."""
    key, model = llm.current_key(), llm.current_model()
    k = (key, model, temperature)
    if k not in _models:
        _models[k] = ChatGoogleGenerativeAI(model=model, google_api_key=key,
                                            temperature=temperature, max_retries=1)
    return _models[k]


def _run(kind, template, schema, temperature, variables):
    def go():
        chain = ChatPromptTemplate.from_template(template) | \
            _chat(temperature).with_structured_output(schema)
        out = chain.invoke(variables)
        if out is None:
            raise RuntimeError("Model returned no structured output")
        return out.model_dump() if hasattr(out, "model_dump") else dict(out)
    return llm._timed(kind, go)  # same latency/success tracing as the SDK backend


class KBRetriever(BaseRetriever):
    """LangChain retriever over the company/role knowledge base
    (ChromaDB with keyword fallback, see kb.py)."""
    company: str = "Generic"
    role: str = ""
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        texts = kb.retrieve(self.company, self.role, query, self.k)
        return [Document(page_content=t, metadata={"company": self.company}) for t in texts]


def retrieve(company, role, query="", k=4):
    return [d.page_content for d in KBRetriever(company=company, role=role, k=k).invoke(query)]


def next_question(resume, role, company, history, mode="normal",
                  prev_topic="", kb_context=None):
    v = llm.question_vars(resume, role, company, history, mode, prev_topic, kb_context)
    mode = v.pop("mode")
    out = _run("question", llm.QUESTION_PROMPT, llm.Question, 0.8, v)
    out["kind"] = mode
    return out


def evaluate(question, answer, role, company, spoken=False):
    v = dict(role=role, company=company, question=question, answer=answer,
             note=llm.SPOKEN_NOTE if spoken else "")
    return llm.clamp_scorecard(_run("evaluate", llm.EVALUATE_PROMPT, llm.Scorecard, 0.2, v))


def judge(question, answer, scorecard):
    v = dict(question=question, answer=answer, scorecard=scorecard)
    return llm.clamp_judgment(_run("judge", llm.JUDGE_PROMPT, llm.Judgment, 0.0, v))


def review_code(title, code, passed, total):
    v = dict(title=title, code=code, passed=passed, total=total)
    return _run("code_review", llm.REVIEW_PROMPT, llm.CodeReview, 0.2, v)
