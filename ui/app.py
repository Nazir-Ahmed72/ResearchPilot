"""ResearchPilot dashboard (Streamlit).

A friendly front end for every API feature. It talks to the running API over HTTP, so start the
API (and the worker) first, then:

    streamlit run ui/app.py

The API address defaults to http://127.0.0.1:8000 (override in the sidebar or with the
RESEARCHPILOT_API_URL environment variable).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
DEFAULT_API = os.environ.get("RESEARCHPILOT_API_URL", "http://127.0.0.1:8000")
UPLOAD_BATCH = 10  # the API accepts up to MAX_FILES_PER_UPLOAD (default 10) files per request

STRATEGIES = {
    "hybrid_rerank": "E4 · Hybrid + rerank (default, best quality)",
    "hybrid": "E3 · Hybrid (fast)",
    "dense": "E1 · Dense / meaning-based",
    "sparse": "E2 · Sparse / keyword (BM25)",
    "hybrid_qt_rerank": "E5 · Hybrid + query rewriting + rerank (uses Groq)",
}
ANALYSIS_TYPES = {
    None: "Detect automatically",
    "factual": "Factual explanation",
    "comparison": "Comparison",
    "limitations": "Limitations",
    "research_gaps": "Research gaps",
    "literature_review": "Literature review",
}
EXAMPLE_QUESTIONS = [
    "",
    "What generator model does RAG use?",
    "Compare traditional RAG and GraphRAG approaches",
    "Compare how RAG and Self-RAG decide when to retrieve passages",
    "What limitations of standard RAG does Self-RAG identify?",
    "What research gaps exist in retrieval-augmented generation?",
    "Write a literature review of retrieval-augmented generation",
    "How many parameters does GPT-4 have?  (not in the papers: should say so)",
]
STEP_LABELS = {
    "understand_query": "1 · Understanding the question",
    "plan_queries": "2 · Planning searches",
    "retrieve": "3 · Searching the papers",
    "rerank": "4 · Ranking the evidence",
    "evaluate_evidence": "5 · Checking the evidence is enough",
    "synthesize": "6 · Writing the answer",
    "verify_citations": "7 · Verifying citations",
    "respond": "8 · Formatting the answer",
}


# ------------------------------------------------------------------------------------- API client


class ApiError(Exception):
    """A readable error: the API's own message, or 'API not reachable'."""


class Api:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base = base_url.rstrip("/") + "/api/v1"
        self.timeout = timeout

    def call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.request(method, self.base + path, **kwargs)
        except httpx.ConnectError as exc:
            raise ApiError(
                "Cannot reach the API. Start it first: `uvicorn app.main:app --port 8000`"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiError("The API did not answer in time; try again.") from exc
        if resp.status_code == 204:
            return None
        try:
            body = resp.json()
        except ValueError:
            body = {"error": {"message": resp.text[:300]}}
        # /health answers 503 with a useful per-service report; everywhere else 503 is an error.
        health_report = resp.status_code == 503 and path.startswith("/health")
        if resp.status_code >= 400 and not health_report:
            err = body.get("error", {}) if isinstance(body, dict) else {}
            message = err.get("message") or f"HTTP {resp.status_code}"
            raise ApiError(message, err.get("details"), resp.status_code)
        return body

    def get(self, path: str, **kw: Any) -> Any:
        return self.call("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> Any:
        return self.call("POST", path, **kw)

    def delete(self, path: str) -> Any:
        return self.call("DELETE", path)


def show_error(exc: ApiError) -> None:
    st.error(str(exc.args[0]))
    if len(exc.args) > 1 and exc.args[1]:
        with st.expander("Details"):
            st.json(exc.args[1])


def upload_files(api: Api, files: list[Any], collection: str | None) -> list[dict[str, Any]]:
    """Send uploaded files in batches; return one result row per file (accepted or not)."""
    rows: list[dict[str, Any]] = []
    for i in range(0, len(files), UPLOAD_BATCH):
        batch = files[i : i + UPLOAD_BATCH]
        multipart = [
            ("files", (f.name.split("/")[-1], f.getvalue(), "application/pdf")) for f in batch
        ]
        data = {"collection": collection} if collection else {}
        try:
            body = api.post("/papers/upload", files=multipart, data=data)
            rows += body["results"]
        except ApiError as exc:  # nothing in this batch was accepted: per-file reasons in details
            details = exc.args[1] if len(exc.args) > 1 else None
            if isinstance(details, list):
                rows += [
                    {
                        "filename": d.get("filename"),
                        "status": d.get("error_code"),
                        "message": d.get("message"),
                        "paper_id": d.get("paper_id"),
                    }
                    for d in details
                ]
            else:
                rows += [
                    {"filename": f.name, "status": "error", "message": str(exc.args[0])}
                    for f in batch
                ]
    return rows


@st.cache_data(ttl=10, show_spinner=False)
def fetch_papers(base_url: str) -> list[dict[str, Any]]:
    api = Api(base_url)
    items: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = api.get("/papers", params={"limit": 100, "offset": offset})
        items += page["items"]
        offset += 100
        if offset >= page["total"]:
            return items


# ------------------------------------------------------------------------------------------ pages


def page_overview(api: Api) -> None:
    st.title("📚 ResearchPilot")
    st.caption("Ask research questions over your papers, with verified citations.")
    try:
        health = api.get("/health")
    except ApiError as exc:
        show_error(exc)
        return
    cols = st.columns(3)
    labels = {"database": "Database (Neon)", "redis": "Queue (Redis)", "qdrant": "Vectors (Qdrant)"}
    for col, (name, label) in zip(cols, labels.items(), strict=True):
        check = health["checks"].get(name, {})
        ok = check.get("status") == "ok"
        col.metric(
            label,
            "✅ OK" if ok else "❌ Down",
            f"{check.get('latency_ms', 0):.0f} ms" if ok else check.get("error"),
        )
    if health["status"] != "ok":
        st.warning(
            "Some services are down. If the database shows a timeout, Neon may be waking "
            "up from sleep; refresh in a few seconds."
        )
    try:
        papers = fetch_papers(api.base.removesuffix("/api/v1"))
    except ApiError:
        papers = []
    ready = sum(p["status"] == "ready" for p in papers)
    c1, c2, c3 = st.columns(3)
    c1.metric("Papers", len(papers))
    c2.metric("Ready to search", ready)
    c3.metric("Chunks indexed", sum(p["chunk_count"] for p in papers))
    st.info(
        "Use the pages on the left: **Papers** to upload, **Search** to find passages, "
        "**Research** to ask questions, **Evaluation** to measure quality."
    )


def page_papers(api: Api) -> None:
    st.title("📄 Papers")
    base = api.base.removesuffix("/api/v1")

    with st.container(border=True):
        st.subheader("Upload")
        mode = st.radio(
            "What do you want to upload?", ["PDF files", "A whole folder"], horizontal=True
        )
        files = st.file_uploader(
            "Choose PDFs" if mode == "PDF files" else "Choose a folder (all PDFs inside are used)",
            type=["pdf"],
            accept_multiple_files=True if mode == "PDF files" else "directory",
        )
        collection = st.text_input(
            "Collection (optional)",
            value="rag-papers",
            help="Groups papers so you can search one group at a time.",
        )
        if st.button("⬆️ Upload", type="primary", disabled=not files):
            with st.spinner(f"Uploading {len(files)} file(s)..."):
                rows = upload_files(api, list(files), collection.strip() or None)
            fetch_papers.clear()
            accepted = sum(r.get("status") == "accepted" for r in rows)
            st.success(
                f"{accepted} of {len(rows)} file(s) accepted. Processing takes about a "
                "minute per paper; refresh the list below."
            )
            st.dataframe(rows, width="stretch", hide_index=True)

    st.subheader("Your papers")
    c1, c2, c3 = st.columns([1, 1, 1])
    status = c1.selectbox("Status", ["all", "ready", "pending", "processing", "failed"])
    if c3.button("🔄 Refresh"):
        fetch_papers.clear()
    try:
        papers = fetch_papers(base)
    except ApiError as exc:
        show_error(exc)
        return
    collections = sorted({p["collection"] for p in papers if p["collection"]})
    coll = c2.selectbox("Collection", ["all", *collections])
    shown = [
        p
        for p in papers
        if (status == "all" or p["status"] == status) and (coll == "all" or p["collection"] == coll)
    ]
    st.dataframe(
        [
            {
                "title": p["title"] or p["original_filename"],
                "status": p["status"],
                "chunks": p["chunk_count"],
                "pages": p["page_count"],
                "year": p["publication_year"],
                "collection": p["collection"],
                "id": p["id"],
            }
            for p in shown
        ],
        width="stretch",
        hide_index=True,
    )
    if not shown:
        return

    st.subheader("Paper details")
    names = {f"{p['title'] or p['original_filename']}  ({p['status']})": p["id"] for p in shown}
    choice = st.selectbox("Choose a paper", list(names))
    paper_id = names[choice]
    try:
        detail = api.get(f"/papers/{paper_id}")
    except ApiError as exc:
        show_error(exc)
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", detail["status"])
    c2.metric("Pages", detail["page_count"])
    c3.metric("Chunks", detail["chunk_count"])
    c4.metric("Year", detail["publication_year"] or "–")
    st.write(f"**Title:** {detail['title']}")
    st.write(
        f"**Authors:** {', '.join(detail['authors']) or '–'}  ·  **DOI:** {detail['doi'] or '–'}"
    )
    if detail["error_message"]:
        st.error(detail["error_message"])
    with st.expander("Processing history"):
        st.dataframe(
            [
                {
                    "status": j["status"],
                    "attempts": j["attempts"],
                    "started": j["started_at"],
                    "finished": j["finished_at"],
                    "error": j["error_message"],
                }
                for j in detail["ingestion_jobs"]
            ],
            width="stretch",
            hide_index=True,
        )

    a, b = st.columns(2)
    if a.button("🔁 Re-process this paper"):
        try:
            api.post(f"/papers/{paper_id}/reingest")
            fetch_papers.clear()
            st.success("Queued for re-processing.")
        except ApiError as exc:
            show_error(exc)
    with b:
        confirm = st.checkbox("I understand deleting is permanent")
        if st.button("🗑️ Delete this paper", disabled=not confirm):
            try:
                api.delete(f"/papers/{paper_id}")
                fetch_papers.clear()
                st.success("Deleted.")
            except ApiError as exc:
                show_error(exc)


def _render_hits(result: dict[str, Any]) -> None:
    timings = " · ".join(f"{k} {v:.0f} ms" for k, v in result["timings_ms"].items())
    st.caption(f"{len(result['results'])} results · {timings}")
    if len(result["queries"]) > 1:
        st.caption("Searched with: " + " | ".join(result["queries"]))
    for hit in result["results"]:
        pages = (
            f"p. {hit['page_start']}"
            if hit["page_start"] == hit["page_end"]
            else f"pp. {hit['page_start']}-{hit['page_end']}"
        )
        with st.expander(
            f"#{hit['rank']} · {hit['title']} · {hit['section']} · {pages} "
            f"· score {hit['score']:.3f}"
        ):
            st.write(hit["text"])
            st.caption(
                "Scores per stage: " + ", ".join(f"{k} {v:.3f}" for k, v in hit["scores"].items())
            )


def page_search(api: Api) -> None:
    st.title("🔎 Search")
    st.caption("Find the passages that match a query, without writing an answer.")
    query = st.text_input("Query", value="How did RAG perform on FEVER fact verification?")
    c1, c2 = st.columns([2, 1])
    strategy = c1.selectbox("Strategy", list(STRATEGIES), format_func=STRATEGIES.get)
    top_k = c2.slider("Results", 1, 20, 5)
    with st.expander("Filters (optional)"):
        f1, f2, f3 = st.columns(3)
        collection = f1.text_input("Collection")
        year_from = f2.number_input("From year", 0, 2100, 0)
        year_to = f3.number_input("To year", 0, 2100, 0)
        sections = st.text_input("Sections (comma-separated, exact names)")
    compare = st.checkbox("Compare all five strategies side by side")

    if st.button("Search", type="primary", disabled=len(query.strip()) < 3):
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if collection.strip():
            body["collection"] = collection.strip()
        if year_from:
            body["year_from"] = int(year_from)
        if year_to:
            body["year_to"] = int(year_to)
        if sections.strip():
            body["sections"] = [s.strip() for s in sections.split(",") if s.strip()]
        chosen = list(STRATEGIES) if compare else [strategy]
        cols = st.columns(len(chosen)) if compare else [st.container()]
        for col, strat in zip(cols, chosen, strict=True):
            with col:
                if compare:
                    st.markdown(f"**{STRATEGIES[strat].split(' · ')[0]}** {strat}")
                with st.spinner("Searching..."):
                    try:
                        _render_hits(
                            api.post("/retrieval/search", json={**body, "strategy": strat})
                        )
                    except ApiError as exc:
                        show_error(exc)


def _wait_for_research(api: Api, request_id: str, timeout_s: int = 600) -> dict[str, Any] | None:
    seen: list[str] = []
    with st.status("Working on your question...", expanded=True) as status:
        start = time.time()
        while time.time() - start < timeout_s:
            try:
                state = api.get(f"/research/{request_id}")
            except ApiError as exc:
                show_error(exc)
                return None
            node = state.get("current_node")
            if node and (not seen or seen[-1] != node):
                seen.append(node)
                st.write(STEP_LABELS.get(node, node))
            if state["status"] not in ("queued", "running"):
                label = {
                    "succeeded": "Done",
                    "failed": "Failed",
                    "interrupted": "Paused for review",
                }.get(state["status"], state["status"])
                status.update(
                    label=label,
                    state="complete" if state["status"] == "succeeded" else "error",
                    expanded=False,
                )
                return state
            if state["status"] == "queued" and time.time() - start > 30 and not seen:
                st.write("⏳ Still queued: is the worker running?")
            time.sleep(3)
        status.update(label="Still running: check again later", state="error")
    return api.get(f"/research/{request_id}")


def _render_research(api: Api, state: dict[str, Any]) -> None:
    st.caption(
        f"Request id: `{state['request_id']}` · type: {state['analysis_type'] or '–'} · "
        f"strategy: {state['strategy']}"
    )
    if state["status"] == "failed":
        st.error(f"The run failed: {state['error_message']}")
        if st.button("▶️ Resume from where it stopped"):
            try:
                api.post(f"/research/{state['request_id']}/resume")
                st.session_state["research_id"] = state["request_id"]
                st.session_state["research_wait"] = True
                st.rerun()
            except ApiError as exc:
                show_error(exc)
        return
    if state["status"] == "interrupted":
        st.warning(f"Paused before '{state['current_node']}' for human review.")
        if st.button("▶️ Continue"):
            api.post(f"/research/{state['request_id']}/resume")
            st.session_state["research_wait"] = True
            st.rerun()
        return
    result = state.get("result")
    if not result:
        st.info(f"Status: {state['status']} ({state['current_node'] or 'waiting'})")
        return

    if result["evidence_sufficient"]:
        st.success("Answered from the papers, with verified citations.")
    else:
        st.warning(
            "The papers do not fully answer this question; see 'Limitations and "
            "uncertainty' in the answer."
        )
    v = result.get("verification") or {}
    m = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sources cited", len(result["references"]))
    c2.metric(
        "Citations valid",
        "–" if v.get("citation_validity") is None else f"{v['citation_validity']:.0%}",
    )
    c3.metric(
        "Claims supported", "–" if v.get("support_rate") is None else f"{v['support_rate']:.0%}"
    )
    c4.metric("Time", f"{m['latency_ms'] / 1000:.0f} s")

    answer, sources, check, metrics = st.tabs(["Answer", "Sources", "Verification", "Metrics"])
    with answer:
        st.markdown(result["answer_markdown"])
    with sources:
        st.dataframe(
            [
                {
                    "#": r["number"],
                    "paper": r["title"],
                    "year": r["year"],
                    "section": r["section"],
                    "pages": f"{r['page_start']}-{r['page_end']}",
                }
                for r in result["references"]
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption("Searched with: " + " | ".join(result["queries"]))
    with check:
        st.json(v)
        if result["removed_statements"]:
            st.write("**Statements removed because their citations did not support them:**")
            for r in result["removed_statements"]:
                st.write(f"- {r['text']}  *(reason: {r['reason']})*")
    with metrics:
        st.write("**Time per step (ms)**")
        st.bar_chart(m["node_timings_ms"], horizontal=True)
        st.write("**Tokens used**", m["token_usage"])
        st.write(f"Retries: {m['retry_count']} · Repairs: {m['repair_count']}")
        if m.get("errors"):
            st.write("**Non-fatal errors**", m["errors"])


def page_research(api: Api) -> None:
    st.title("🧠 Research")
    st.caption(
        "Ask a question; the answer cites the exact papers and pages it is based on. "
        "Each question uses about 7,000 Groq tokens."
    )
    example = st.selectbox("Example questions (optional)", EXAMPLE_QUESTIONS)
    question = st.text_area("Your question", value=example.split("  (")[0], height=90)
    c1, c2, c3 = st.columns(3)
    analysis = c1.selectbox("Analysis type", list(ANALYSIS_TYPES), format_func=ANALYSIS_TYPES.get)
    strategy = c2.selectbox("Retrieval", list(STRATEGIES), format_func=STRATEGIES.get)
    collection = c3.text_input("Collection (optional)", value="rag-papers")
    try:
        papers = [
            p for p in fetch_papers(api.base.removesuffix("/api/v1")) if p["status"] == "ready"
        ]
    except ApiError:
        papers = []
    titles = {(p["title"] or p["original_filename"])[:90]: p["id"] for p in papers}
    picked = st.multiselect("Only these papers (optional)", list(titles))

    if st.button("Ask", type="primary", disabled=len(question.strip()) < 5):
        body: dict[str, Any] = {"question": question.strip(), "strategy": strategy}
        if analysis:
            body["analysis_type"] = analysis
        if collection.strip():
            body["collection"] = collection.strip()
        if picked:
            body["paper_ids"] = [titles[t] for t in picked]
        try:
            created = api.post("/research/query", json=body)
            st.session_state["research_id"] = created["request_id"]
            st.session_state["research_wait"] = True
        except ApiError as exc:
            show_error(exc)

    with st.expander("Open an earlier request by id"):
        rid = st.text_input("Request id")
        if st.button("Open") and rid.strip():
            st.session_state["research_id"] = rid.strip()
            st.session_state["research_wait"] = False

    request_id = st.session_state.get("research_id")
    if not request_id:
        return
    st.divider()
    try:
        if st.session_state.pop("research_wait", False):
            state = _wait_for_research(api, request_id)
        else:
            state = api.get(f"/research/{request_id}")
    except ApiError as exc:
        show_error(exc)
        return
    if state:
        _render_research(api, state)


def page_evaluation(api: Api) -> None:
    st.title("📊 Evaluation")
    st.caption("Measure retrieval quality on the 20 labelled questions (runs in the worker).")
    with st.container(border=True):
        st.subheader("Start an evaluation")
        c1, c2 = st.columns(2)
        name = c1.text_input("Run name (letters, numbers, - and _)", value="dashboard-eval")
        top_k = c2.slider("Top-K", 1, 20, 5)
        experiments = st.multiselect(
            "Retrieval experiments",
            ["E1", "E2", "E3", "E4", "E5"],
            default=["E1", "E2", "E3", "E4"],
            help="E1 dense · E2 BM25 · E3 hybrid · E4 hybrid+rerank · E5 adds query rewriting "
            "(uses Groq). E4/E5 take ~3 minutes each on CPU.",
        )
        generation = st.checkbox(
            "Also run the full research workflow on every question "
            "(⚠️ uses most of a day's Groq quota)"
        )
        ragas = st.checkbox(
            "Add Ragas scores (needs the option above; more Groq usage)", disabled=not generation
        )
        langsmith = st.checkbox("Publish to LangSmith as a dataset + experiments")
        if st.button("▶️ Start", type="primary", disabled=not (experiments or generation)):
            body = {
                "name": name,
                "experiments": experiments,
                "top_k": top_k,
                "include_generation": generation,
                "ragas": ragas and generation,
                "pace_seconds": 30 if generation else 0,
                "langsmith": langsmith,
            }
            try:
                started = api.post("/evaluations/run", json=body)
                st.success(
                    f"Started batch {started['batch_id']} ({len(started['runs'])} runs). "
                    "Refresh the table below in a few minutes."
                )
            except ApiError as exc:
                show_error(exc)

    st.subheader("Results")
    if st.button("🔄 Refresh results"):
        pass
    try:
        runs = api.get("/evaluations/results", params={"limit": 50})["items"]
    except ApiError as exc:
        show_error(exc)
        runs = []
    if runs:
        rows = []
        for r in runs:
            m = r["metrics"] or {}
            recall = next((v for k, v in m.items() if k.startswith("recall@")), None)
            rows.append(
                {
                    "name": r["name"],
                    "experiment": r["experiment"],
                    "strategy": r["retrieval_strategy"],
                    "status": r["status"],
                    "recall": recall,
                    "mrr": m.get("mrr"),
                    "ndcg": next((v for k, v in m.items() if k.startswith("ndcg@")), None),
                    "latency ms": m.get("latency_ms_mean"),
                    "created": r["created_at"][:19],
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No evaluation runs yet.")

    st.subheader("Saved reports")
    folders = (
        sorted((p for p in RESULTS_DIR.glob("*") if (p / "report.md").exists()), reverse=True)
        if RESULTS_DIR.exists()
        else []
    )
    if not folders:
        st.info("No reports in experiments/results yet.")
        return
    folder = st.selectbox("Report", folders, format_func=lambda p: p.name)
    st.markdown((folder / "report.md").read_text(encoding="utf-8"))


WORKER_CMD = (
    "celery -A app.workers.celery_app worker -Q ingestion,research --pool=solo "
    "--loglevel=INFO --without-gossip --without-mingle --without-heartbeat"
)


def page_help() -> None:
    st.title("❓ How to run")
    st.markdown(
        f"""
**Start these first, each in its own terminal (venv activated):**

1. Worker (processes uploads and questions):
   ```
   {WORKER_CMD}
   ```
2. API:
   ```
   uvicorn app.main:app --port 8000
   ```
3. This dashboard:
   ```
   streamlit run ui/app.py
   ```

**Pages**
* **Overview:** are the services up? How many papers are ready?
* **Papers:** upload files or a whole folder, watch them become *ready*, re-process or delete.
* **Search:** see which passages each retrieval strategy finds (no answer is written).
* **Research:** ask a question and get a cited answer; watch the steps live.
* **Evaluation:** measure retrieval quality and read saved reports.

**If something fails**
* *Cannot reach the API*: start `uvicorn` (step 2).
* *Question stays queued*: start the worker (step 1).
* *Rate limit / 429*: Groq's free daily quota is used up; try again later and use **Resume**.
        """
    )


# ------------------------------------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="ResearchPilot", page_icon="📚", layout="wide")
    with st.sidebar:
        st.header("📚 ResearchPilot")
        page = st.radio(
            "Go to",
            ["🏠 Overview", "📄 Papers", "🔎 Search", "🧠 Research", "📊 Evaluation", "❓ Help"],
            label_visibility="collapsed",
        )
        st.divider()
        base_url = st.text_input("API address", value=DEFAULT_API)
        api = Api(base_url)
        try:
            api.get("/health/live")
            st.success("API connected")
        except ApiError:
            st.error("API not running")
        st.markdown(f"[Open API docs]({base_url.rstrip('/')}/docs)")

    if page == "🏠 Overview":
        page_overview(api)
    elif page == "📄 Papers":
        page_papers(api)
    elif page == "🔎 Search":
        page_search(api)
    elif page == "🧠 Research":
        page_research(api)
    elif page == "📊 Evaluation":
        page_evaluation(api)
    else:
        page_help()


main()
