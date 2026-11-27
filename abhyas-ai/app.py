import os
import uuid

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pypdf import PdfReader

import db
import interview_graph
import kb
import llm
import sandbox
import voice

st.set_page_config(page_title="ABHYAS.AI", page_icon="🎯", layout="wide")
db.init()
ss = st.session_state
for k, v in dict(started=False, q=None, idx=0, result=None, extra=None, sid=None,
                 history=[], done=False, total=5, ctx=[], tts={},
                 owner=uuid.uuid4().hex).items():
    ss.setdefault(k, v)

METRICS = ["technical_accuracy", "clarity", "communication", "depth"]


def explain(msg):
    """Friendly error display for LLM failures."""
    msg = str(msg or "unknown error")
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        st.error("⏳ Gemini ki free quota is model ke liye khatam ho gayi hai. "
                 "Sidebar me **Gemini model** badal ke dekho (jaise gemini-2.5-flash-lite), "
                 "ya quota reset hone ka wait karo (daily limit midnight Pacific time par reset hoti hai).")
    else:
        st.error(f"LLM call failed: {msg}")


def call(fn, *a, **kw):
    """Run an LLM call; show a friendly error instead of crashing the app."""
    try:
        return fn(*a, **kw)
    except Exception as e:
        explain(e)
        st.stop()


def be():
    """LLM backend module chosen in the sidebar (LangChain or direct SDK)."""
    return interview_graph.BACKENDS[BACKEND]()


def secret_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def radar(values, labels):
    fig = go.Figure(go.Scatterpolar(r=list(values) + [values[0]],
                                    theta=list(labels) + [labels[0]], fill="toself"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10])), showlegend=False,
                      margin=dict(t=20, b=20))
    return fig


# ================= sidebar =================
with st.sidebar:
    st.header("⚙️ Setup")
    server_key = os.getenv("GEMINI_API_KEY", "") or secret_key()
    key_in = st.text_input("Gemini API key", type="password",
                           help="Free key: aistudio.google.com. your key is used only in this session.")
    key = key_in.strip() or server_key
    llm.set_key(key)
    if server_key and not key_in.strip():
        st.caption("🔑 Server key active (shared quota). ")
    model_name = st.text_input(
        "Gemini model", value=os.getenv("GEMINI_MODEL", llm.MODEL),
        help="Quota har model ki alag hoti hai. 429 error aaye to gemini-2.5-flash-lite try karo.")
    llm.set_model(model_name)
    engine = st.selectbox("LLM engine", ["LangChain", "Direct Gemini SDK"],
                          help="Interview flow dono me LangGraph se chalta hai. "
                               "Direct SDK sirf fallback/baseline hai.")
    BACKEND = "langchain" if engine == "LangChain" else "sdk"
    CFG = {"configurable": {"api_key": key, "model": model_name}}
    role = st.text_input("Target role", "Data Analyst")
    company = st.selectbox("Company style (RAG)", kb.companies(),
                           index=kb.companies().index("Generic"))
    ss.total = st.slider("Number of questions", 3, 10, 5)
    st.checkbox("🔊 Read questions aloud (voice-to-voice)", key="voice_q")
    st.checkbox("⚖️ Run LLM-judge on every answer (extra API call)", key="judge_live")
    up = st.file_uploader("Resume (PDF)", type="pdf")
    if st.button("🚀 Start interview", type="primary"):
        if not key or not up:
            st.error("Required API Key and Resume both.")
        else:
            text = "\n".join(p.extract_text() or "" for p in PdfReader(up).pages)
            if len(text.strip()) < 50:
                st.error("Resume se text nahi nikla (scanned PDF?).")
            else:
                gs = interview_graph.start_state(text, role, company, ss.total, BACKEND)
                with st.spinner("Preparing questions..."):
                    gs = call(interview_graph.GRAPH.invoke, gs, CFG)
                if not gs.get("current_q"):
                    explain(gs.get("ask_error"))
                    st.stop()
                ss.update(resume=text, role=role, company=company, idx=0, history=[],
                          result=None, extra=None, done=False, started=True, tts={},
                          sid=db.new_session(role, company, ss.owner),
                          gstate=gs, q=gs["current_q"], ctx=gs.get("kb_context", []))
                st.rerun()

tab_int, tab_code, tab_dash, tab_eval = st.tabs(
    ["🎙️ Mock Interview", "💻 Code Arena", "📈 Insights", "🧪 AI Quality Lab"])


# ================= interview =================
def finish_answer(q, answer, extra):
    """One LangGraph turn: evaluate -> (judge) -> plan -> retrieve -> ask next question."""
    state = {**ss.gstate, "answer": answer, "spoken": extra["mode"] == "voice",
             "judge_on": bool(ss.get("judge_live")), "total": ss.total, "backend": BACKEND}
    with st.spinner("Evaluating and preparing next question..."):
        new = call(interview_graph.GRAPH.invoke, state, CFG)
    entry = new["history"][-1]
    aid = db.save_answer(ss.sid, q, answer, new["scorecard"], entry["avg"], extra)
    if entry.get("judgment"):
        db.save_judgment(aid, entry["judgment"])
    ss.history.append({**entry, **extra})
    ss.gstate, ss.result, ss.extra = new, new["scorecard"], extra
    st.rerun()


def report_md():
    lines = [f"# Abhyas AI: Interview report: {ss.role} ({ss.company})\n"]
    for i, h in enumerate(ss.history, 1):
        lines += [f"## Q{i}. {h['question']}", f"*{h['topic']} / {h['difficulty']}* "
                  f"— score {h['avg']:.1f}/10\n", f"**Your answer:** {h['answer']}\n",
                  f"**Feedback:** {h['feedback']}\n", f"**Ideal answer:** {h['ideal_answer']}\n"]
    return "\n".join(lines)


with tab_int:
    if not ss.started:
        st.markdown("<h1 style='text-align:center; margin-bottom:0'>🎯 ABHYAS: AI Learning Companion</h1>",
                    unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:gray; font-size:1.05rem'>"
                    "AI Interview Coach · Practice Powered by Intelligence.</p>",
                    unsafe_allow_html=True)
        st.markdown("<p style='text-align:center'>Upload resume on sidebar and select role and company"
                    "</p>",
                    unsafe_allow_html=True)
    elif ss.done:
        st.success("Interview complete! 🎉")
        df = pd.DataFrame(ss.history)
        avg = df[METRICS].mean()
        a, b = st.columns([1, 2])
        a.metric("Overall score", f"{avg.mean():.1f} / 10")
        v = df[df["mode"] == "voice"]
        if not v.empty:
            a.metric("Avg filler words / answer", f"{v['fillers'].mean():.1f}")
            a.metric("Avg pace (wpm)", f"{v['wpm'].dropna().mean():.0f}")
        b.plotly_chart(radar(avg.tolist(), METRICS), width="stretch", key="pc_summary")
        weak = sorted({w for h in ss.history for w in h["weak_points"]})
        if weak:
            st.write("**Improve karo:** " + ", ".join(weak))
        st.download_button("⬇️ Download report (.md)", report_md(), "interview_report.md")
    else:
        q = ss.q
        st.progress(ss.idx / ss.total, text=f"Question {ss.idx + 1} of {ss.total}")
        st.caption(f"Topic: {q['topic']} · Difficulty: {q['difficulty']}")
        st.subheader(q["question"])
        with st.expander("📚 Company context retrieved by RAG"):
            for c in ss.ctx:
                st.write("• " + c)

        if ss.get("voice_q"):
            if q["question"] not in ss.tts:
                ss.tts[q["question"]] = voice.tts(q["question"])
            if ss.tts[q["question"]]:
                st.audio(ss.tts[q["question"]], format="audio/mp3")

        if ss.result is None:
            modes = ["⌨️ Text"] + (["🎙️ Voice"] if hasattr(st, "audio_input") else [])
            mode = st.radio("Answer mode", modes, horizontal=True)
            n = len(ss.history)
            if mode.startswith("⌨️"):
                ans = st.text_area("Your answer", height=180, key=f"t{n}")
                if st.button("Submit answer", type="primary"):
                    finish_answer(q, ans, dict(mode="text", wpm=None, fillers=None, words=None))
            else:
                rec = st.audio_input("Record your answer", key=f"a{n}")
                if rec and st.button("Submit recording", type="primary"):
                    data = rec.getvalue()
                    with st.spinner("Transcribing..."):
                        text = call(llm.transcribe, data)
                    stats = voice.analyze(text, voice.wav_duration(data))
                    finish_answer(q, text, dict(mode="voice", wpm=stats["wpm"],
                                                fillers=stats["fillers"], words=stats["words"],
                                                breakdown=stats["filler_breakdown"],
                                                pace=stats["pace_label"]))
        else:
            s, ex = ss.result, ss.extra
            avg = sum(s[m] for m in METRICS) / 4
            for c, m in zip(st.columns(4), METRICS):
                c.metric(m.replace("_", " ").title(), f"{s[m]}/10")
            if ex["mode"] == "voice":
                st.info(f"🎙️ **Transcript:** {ss.history[-1]['answer']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Pace", f"{ex['wpm']} wpm" if ex["wpm"] else "n/a", ex["pace"])
                c2.metric("Filler words", ex["fillers"])
                c3.metric("Words", ex["words"])
                if ex["breakdown"]:
                    st.caption("Fillers: " + ", ".join(f"{k} ×{v}" for k, v in ex["breakdown"].items()))
            st.write(f"**Feedback:** {s['feedback']}")
            if s["weak_points"]:
                st.write("**Weak points:** " + ", ".join(s["weak_points"]))
            with st.expander("💡 Ideal answer vs your answer", expanded=True):
                l, r = st.columns(2)
                l.markdown("**Your answer**")
                l.write(ss.history[-1]["answer"] or "(empty)")
                r.markdown("**Ideal answer**")
                r.write(s["ideal_answer"])

            gs = ss.gstate
            if not gs.get("done") and not gs.get("current_q"):
                st.warning("Agla sawal abhi generate nahi hua (quota/network?). "
                           "Next dabane par dobara try hoga. Tumhara scorecard safe hai.")
            if st.button("Next ➡️"):
                if gs.get("done") or ss.idx + 1 >= ss.total:
                    ss.idx += 1
                    ss.result = None
                    ss.done = True
                else:
                    if not gs.get("current_q"):  # earlier attempt failed: retry now
                        with st.spinner("Agla sawal..."):
                            gs = call(interview_graph.GRAPH.invoke,
                                      {**gs, "answer": None, "total": ss.total,
                                       "backend": BACKEND}, CFG)
                        if not gs.get("current_q"):
                            explain(gs.get("ask_error"))
                            st.stop()
                        ss.gstate = gs
                    ss.idx += 1
                    ss.result = None
                    ss.q, ss.ctx = gs["current_q"], gs.get("kb_context", [])
                st.rerun()

# ================= code lab =================
with tab_code:
    st.title("💻 Code Arena")
    st.caption("SQL: in-memory SQLite (read-only). Python: runs in a subprocess with a 5s timeout.")
    kind = st.radio("Type", ["SQL", "Python"], horizontal=True)
    chs = sandbox.SQL_CHALLENGES if kind == "SQL" else sandbox.PY_CHALLENGES
    ch = st.selectbox("Challenge", chs, format_func=lambda c: c["title"])
    st.write(ch["prompt"])
    if kind == "SQL":
        st.code(sandbox.SQL_TABLES_DOC, language="text")
    code = st.text_area("Your code", value="" if kind == "SQL" else ch["starter"],
                        height=200, key=f"code_{ch['id']}")
    if st.button("▶ Run & review", type="primary"):
        res = (sandbox.run_sql(code, ch) if kind == "SQL" else sandbox.run_python(code, ch))
        db.save_code_attempt(ch["id"], kind, res["passed"], res["total"], ss.owner)
        if res["error"]:
            st.error(res["error"])
        elif res["passed"] == res["total"]:
            st.success(f"✅ All tests passed ({res['passed']}/{res['total']})")
        else:
            st.warning(f"Passed {res['passed']}/{res['total']}")
        if kind == "SQL" and res.get("rows") is not None:
            st.dataframe(pd.DataFrame(res["rows"], columns=res["columns"]))
        if res["ok"] and key:
            with st.spinner("AI review..."):
                rv = call(be().review_code, ch["title"], code, res["passed"], res["total"])
            st.write(f"**Time:** {rv['time_complexity']} · **Space:** {rv['space_complexity']}")
            st.write(rv["feedback"])
            st.info("💡 " + rv["improvement"])

# ================= dashboard =================
with tab_dash:
    st.title("📈 Insights Dashboard")
    df_all = db.all_answers(ss.owner)
    if df_all.empty:
        st.info("Abhi koi data nahi. Pehle ek mock interview do.")
    else:
        df_all["mode"] = df_all["mode"].fillna("text")
        with st.expander("🔎 Filters", expanded=True):
            f1, f2, f3 = st.columns(3)
            comps = f1.multiselect("Company", sorted(df_all["company"].unique()),
                                   default=sorted(df_all["company"].unique()), key="f_comp")
            roles = f2.multiselect("Role", sorted(df_all["role"].unique()),
                                   default=sorted(df_all["role"].unique()), key="f_role")
            modes = f3.multiselect("Answer mode", ["text", "voice"],
                                   default=["text", "voice"], key="f_mode")
            lo, hi = st.slider("Score range", 0.0, 10.0, (0.0, 10.0), 0.5, key="f_score")
        df = df_all[df_all["company"].isin(comps) & df_all["role"].isin(roles)
                    & df_all["mode"].isin(modes) & df_all["avg"].between(lo, hi)]

        if df.empty:
            st.warning("In filters ke saath koi data nahi mila. Filters badlo.")
        else:
            sess = df.groupby("session_id")["avg"].mean()
            delta = (sess.iloc[-1] - sess.iloc[-2]) if len(sess) > 1 else None
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Sessions", len(sess))
            k2.metric("Answers", len(df))
            k3.metric("Avg score", f"{df['avg'].mean():.1f} / 10")
            k4.metric("Latest session", f"{sess.iloc[-1]:.1f}",
                      f"{delta:+.1f}" if delta is not None else None)
            st.download_button("⬇️ Export filtered data (CSV)", df.to_csv(index=False),
                               "abhyas_data.csv", key="d_csv")

            t_over, t_topic, t_expl, t_speak, t_code = st.tabs(
                ["Overview", "Topics & skill gaps", "Answer explorer", "Speaking", "Code"])

            with t_over:
                ready = df.sort_values("id").tail(15)["avg"].mean() * 10
                g = go.Figure(go.Indicator(
                    mode="gauge+number", value=round(ready),
                    title={"text": "Readiness (last 15 answers)"},
                    gauge={"axis": {"range": [0, 100]},
                           "steps": [{"range": [0, 40], "color": "#7f1d1d"},
                                     {"range": [40, 70], "color": "#854d0e"},
                                     {"range": [70, 100], "color": "#14532d"}]}))
                g.update_layout(height=300, margin=dict(t=60, b=10))
                c1, c2 = st.columns(2)
                c1.plotly_chart(g, width="stretch", key="d_gauge")
                c2.plotly_chart(radar(df[METRICS].mean().tolist(), METRICS),
                                width="stretch", key="d_radar")
                st.caption("Readiness is a rough estimate from recent AI scores, not a prediction.")
                sel = st.multiselect("Metrics to plot over sessions", METRICS,
                                     default=METRICS, key="d_metrics")
                if sel:
                    tr = df.groupby("session_id")[sel].mean().reset_index().melt(
                        "session_id", var_name="metric", value_name="score")
                    st.plotly_chart(px.line(tr, x="session_id", y="score", color="metric",
                                            markers=True, range_y=[0, 10],
                                            title="Progress across sessions"),
                                    width="stretch", key="d_trend")

            with t_topic:
                heat = df.groupby("topic")[METRICS].mean()
                st.plotly_chart(px.imshow(heat, text_auto=".1f", aspect="auto", zmin=0, zmax=10,
                                          color_continuous_scale="RdYlGn",
                                          title="Topic x skill heatmap (red = weak)"),
                                width="stretch", key="d_heat")
                tp = df.groupby("topic").agg(avg=("avg", "mean"),
                                             answers=("id", "count")).sort_values("avg").reset_index()
                st.plotly_chart(px.bar(tp, x="avg", y="topic", orientation="h", color="avg",
                                       range_x=[0, 10], color_continuous_scale="RdYlGn",
                                       hover_data=["answers"],
                                       title="Topic-wise average (lowest = weakest)"),
                                width="stretch", key="d_topics")
                st.success("🎯 **Next focus:** " + ", ".join(tp["topic"].head(3)))

            with t_expl:
                e1, e2 = st.columns(2)
                tsel = e1.multiselect("Topic", sorted(df["topic"].unique()), key="e_topic")
                dsel = e2.multiselect("Difficulty", sorted(df["difficulty"].dropna().unique()),
                                      key="e_diff")
                ex = df.copy()
                if tsel:
                    ex = ex[ex["topic"].isin(tsel)]
                if dsel:
                    ex = ex[ex["difficulty"].isin(dsel)]
                ex = ex.sort_values("avg")
                st.dataframe(ex[["id", "created", "topic", "difficulty", "question", "avg"]],
                             hide_index=True, width="stretch",
                             column_config={"avg": st.column_config.ProgressColumn(
                                 "Score", min_value=0, max_value=10, format="%.1f")})
                if not ex.empty:
                    pick = st.selectbox("Open an answer", ex["id"].tolist(), key="e_pick",
                                        format_func=lambda i: ex.loc[ex["id"] == i, "question"]
                                        .iloc[0][:80])
                    r = ex[ex["id"] == pick].iloc[0]
                    st.markdown(f"**Q:** {r['question']}")
                    st.markdown(f"**Your answer:** {r['answer'] or '(empty)'}")
                    for c, m in zip(st.columns(4), METRICS):
                        c.metric(m.replace("_", " ").title(), f"{r[m]:.0f}/10")

            with t_speak:
                vd = df[df["mode"] == "voice"].reset_index(drop=True)
                if vd.empty:
                    st.info("Voice mode me jawab do, tab speaking analytics yahan dikhega.")
                else:
                    s1, s2 = st.columns(2)
                    s1.metric("Avg pace", f"{vd['wpm'].dropna().mean():.0f} wpm")
                    s2.metric("Avg fillers / answer", f"{vd['fillers'].mean():.1f}")
                    fw = px.line(vd, y="wpm", markers=True, title="Pace per answer (wpm)")
                    fw.add_hrect(y0=120, y1=160, fillcolor="green", opacity=0.15,
                                 annotation_text="ideal 120-160")
                    st.plotly_chart(fw, width="stretch", key="d_wpm")
                    st.plotly_chart(px.bar(vd, y="fillers", title="Filler words per answer"),
                                    width="stretch", key="d_fill")
                    st.plotly_chart(px.scatter(vd, x="fillers", y="avg", range_y=[0, 10],
                                               title="Do filler words affect your score?"),
                                    width="stretch", key="d_scatter")

            with t_code:
                cd = db.code_df(ss.owner)
                if cd.empty:
                    st.info("Code Arena me kuch challenges solve karo.")
                else:
                    cd["solved"] = cd["passed"] == cd["total"]
                    st.write(f"Attempts: {len(cd)} · Solved: {int(cd['solved'].sum())} · "
                             f"Pass rate: {cd['solved'].mean():.0%}")
                    per = cd.groupby("challenge")["solved"].mean().reset_index()
                    st.plotly_chart(px.bar(per, x="challenge", y="solved", range_y=[0, 1],
                                           title="Pass rate per challenge"),
                                    width="stretch", key="d_code")

# ================= AI quality =================
with tab_eval:
    st.title("🧪 AI Quality Lab")
    with st.expander("🧭 How the interview workflow works (LangGraph)"):
        st.markdown(
            "- **evaluate**: LangChain chain scores the answer (Pydantic scorecard)\n"
            "- **judge** (optional): a second chain audits the evaluator\n"
            "- **plan**: weak answer -> cross-question, strong -> harder, never two deep-dives in a row\n"
            "- **retrieve**: LangChain retriever pulls company context from the knowledge base\n"
            "- **ask**: generates the next question using the coverage plan\n"
            "- **finish**: ends after the last answer")
        st.code(interview_graph.mermaid(), language="text")
    st.caption("A second LLM call audits the evaluator (fairness, feedback quality, grounding). "
               "Every LLM call's latency and success is logged to SQLite.")
    if st.button("Judge latest unjudged answers (max 10)"):
        todo = db.unjudged(10, ss.owner)
        if todo.empty:
            st.info("Sab answers already judged.")
        for _, r in todo.iterrows():
            sc = {m: r[m] for m in METRICS}
            db.save_judgment(int(r["id"]), call(be().judge, r["question"], r["answer"], sc))
        st.rerun()
    j = db.judgments_df(ss.owner)
    if j.empty:
        st.info("Abhi koi judgment nahi.")
    else:
        a, b, c = st.columns(3)
        a.metric("Fairness", f"{j['fairness'].mean():.1f}/10")
        b.metric("Feedback quality", f"{j['feedback_quality'].mean():.1f}/10")
        c.metric("Grounded", f"{j['grounded'].mean():.1f}/10")
        st.dataframe(j[["answer_id", "fairness", "feedback_quality", "grounded", "comment"]])
    calls = db.calls_df()
    if not calls.empty:
        st.subheader("Latency by call type")
        g = calls.groupby("kind").agg(calls=("id", "count"),
                                      p50_ms=("latency_ms", "median"),
                                      p95_ms=("latency_ms", lambda x: x.quantile(0.95)),
                                      success=("ok", "mean")).round(0)
        st.dataframe(g)
