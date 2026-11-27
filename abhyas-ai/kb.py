"""Company/role knowledge base with RAG.
Primary: ChromaDB (persistent, default embeddings). Fallback: keyword overlap,
so the app still works if Chroma/embedding download fails."""
import json
import pathlib
import re

KB_PATH = pathlib.Path(__file__).parent / "kb" / "question_bank.json"
_col = None


def load():
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def companies():
    return sorted({e["company"] for e in load()})


def _collection():
    global _col
    if _col is not None:
        return _col
    try:
        import chromadb
        col = chromadb.PersistentClient(path=".chroma").get_or_create_collection("interview_kb")
        items = load()
        if col.count() != len(items):
            col.upsert(ids=[e["id"] for e in items],
                       documents=[e["text"] for e in items],
                       metadatas=[{"company": e["company"], "type": e["type"]} for e in items])
        _col = col
    except Exception:
        _col = False
    return _col


def _keyword(company, query, k):
    q = set(re.findall(r"[a-z]+", query.lower()))
    pool = [e for e in load() if e["company"] in (company, "Generic")]
    pool.sort(key=lambda e: (e["company"] == company,
                             len(q & set(re.findall(r"[a-z]+", e["text"].lower())))),
              reverse=True)
    return [e["text"] for e in pool[:k]]


def retrieve(company, role, extra="", k=4):
    query = f"{role} {company} interview {extra}".strip()
    col = _collection()
    if col:
        try:
            r = col.query(query_texts=[query], n_results=k,
                          where={"company": {"$in": [company, "Generic"]}})
            docs = r["documents"][0]
            if docs:
                return docs
        except Exception:
            pass
    return _keyword(company, query, k)
