import sqlite3
import pandas as pd

DB = "coach.db"


def _conn():
    return sqlite3.connect(DB)


def init():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS sessions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT DEFAULT CURRENT_TIMESTAMP, role TEXT, company TEXT);
        CREATE TABLE IF NOT EXISTS answers(
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
            topic TEXT, difficulty TEXT, question TEXT, answer TEXT,
            technical_accuracy REAL, clarity REAL, communication REAL, depth REAL, avg REAL);
        CREATE TABLE IF NOT EXISTS llm_calls(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT DEFAULT CURRENT_TIMESTAMP, kind TEXT, latency_ms REAL, ok INTEGER);
        CREATE TABLE IF NOT EXISTS judgments(
            id INTEGER PRIMARY KEY AUTOINCREMENT, answer_id INTEGER,
            fairness REAL, feedback_quality REAL, grounded REAL, comment TEXT);
        CREATE TABLE IF NOT EXISTS code_attempts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT DEFAULT CURRENT_TIMESTAMP,
            challenge TEXT, kind TEXT, passed INTEGER, total INTEGER);
        """)
        for tbl in ("sessions", "code_attempts"):  # per-visitor isolation
            tcols = {r[1] for r in c.execute(f"PRAGMA table_info({tbl})")}
            if "owner" not in tcols:
                c.execute(f"ALTER TABLE {tbl} ADD COLUMN owner TEXT DEFAULT 'local'")
        cols = {r[1] for r in c.execute("PRAGMA table_info(answers)")}
        for name, typ in [("mode", "TEXT"), ("wpm", "REAL"),
                          ("fillers", "INTEGER"), ("words", "INTEGER")]:
            if name not in cols:  # migrate older DBs
                c.execute(f"ALTER TABLE answers ADD COLUMN {name} {typ}")


def new_session(role, company, owner="local"):
    with _conn() as c:
        return c.execute("INSERT INTO sessions(role, company, owner) VALUES(?,?,?)",
                         (role, company, owner)).lastrowid


def save_answer(sid, q, answer, s, avg, extra=None):
    extra = extra or {}
    with _conn() as c:
        return c.execute(
            """INSERT INTO answers(session_id, topic, difficulty, question, answer,
               technical_accuracy, clarity, communication, depth, avg,
               mode, wpm, fillers, words) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, q["topic"], q["difficulty"], q["question"], answer,
             s["technical_accuracy"], s["clarity"], s["communication"], s["depth"], avg,
             extra.get("mode", "text"), extra.get("wpm"), extra.get("fillers"),
             extra.get("words"))).lastrowid


def all_answers(owner="local"):
    with _conn() as c:
        return pd.read_sql_query(
            """SELECT a.*, s.created, s.role, s.company FROM answers a
               JOIN sessions s ON s.id = a.session_id WHERE s.owner = ?""", c, params=(owner,))


def log_call(kind, latency_ms, ok):
    with _conn() as c:
        c.execute("INSERT INTO llm_calls(kind, latency_ms, ok) VALUES(?,?,?)",
                  (kind, latency_ms, int(ok)))


def calls_df():
    with _conn() as c:
        return pd.read_sql_query("SELECT * FROM llm_calls", c)


def unjudged(limit=10, owner="local"):
    with _conn() as c:
        return pd.read_sql_query(
            """SELECT a.* FROM answers a JOIN sessions s ON s.id = a.session_id
               WHERE s.owner = ? AND a.id NOT IN (SELECT answer_id FROM judgments)
               ORDER BY a.id DESC LIMIT ?""", c, params=(owner, limit))


def save_judgment(answer_id, j):
    with _conn() as c:
        c.execute("""INSERT INTO judgments(answer_id, fairness, feedback_quality, grounded, comment)
                     VALUES(?,?,?,?,?)""",
                  (answer_id, j["fairness"], j["feedback_quality"], j["grounded"], j["comment"]))


def judgments_df(owner="local"):
    with _conn() as c:
        return pd.read_sql_query(
            """SELECT j.* FROM judgments j JOIN answers a ON a.id = j.answer_id
               JOIN sessions s ON s.id = a.session_id WHERE s.owner = ?""", c, params=(owner,))


def save_code_attempt(challenge, kind, passed, total, owner="local"):
    with _conn() as c:
        c.execute("INSERT INTO code_attempts(challenge, kind, passed, total, owner) VALUES(?,?,?,?,?)",
                  (challenge, kind, passed, total, owner))


def code_df(owner="local"):
    with _conn() as c:
        return pd.read_sql_query("SELECT * FROM code_attempts WHERE owner = ?", c, params=(owner,))
