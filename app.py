"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from common.schemas import AUTO_APPROVE_THRESHOLD, ESCALATE_THRESHOLD
from exercises.exercise_4_audit import build_graph


load_dotenv()


def _new_thread_id() -> str:
    import uuid

    return str(uuid.uuid4())


def _analysis_field(analysis, name: str):
    """Read a field from a PRAnalysis model or a checkpoint dict."""
    if analysis is None:
        return None
    if isinstance(analysis, dict):
        return analysis.get(name)
    return getattr(analysis, name, None)


async def _recent_threads() -> list[dict]:
    """Same grouping as `python -m audit.replay --list`."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp) AS started,
                   MAX(timestamp) AS last_event,
                   MAX(risk_level) AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 25
            """
        ) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def _load_checkpoint(thread_id: str):
    """Return (state values, interrupt payload or None) for a saved thread."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        snapshot = await app.aget_state({"configurable": {"thread_id": thread_id}})
        payload = snapshot.interrupts[0].value if snapshot.interrupts else None
        return snapshot.values or {}, payload


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


def _open_recent_session() -> None:
    """ButtonColumn callback: jump back to a saved review thread."""
    click = st.session_state.get("session_open")
    if not click:
        return
    rows = st.session_state.get("recent_rows") or []
    row_index = click["row"] if isinstance(click, dict) else click.row
    if row_index is None or row_index >= len(rows):
        return
    row = rows[row_index]
    st.session_state.thread_id = row["thread_id"]
    st.session_state.pr_url = row["pr_url"]
    st.session_state.pr_url_input = row["pr_url"]
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.load_thread = True


# ─── Session state ─────────────────────────────────────────────────────────
st.session_state.setdefault("thread_id", None)
st.session_state.setdefault("pr_url", "")
st.session_state.setdefault("pr_url_input", "")
st.session_state.setdefault("interrupt_payload", None)
st.session_state.setdefault("final", None)
st.session_state.setdefault("load_thread", False)
st.session_state.setdefault("recent_rows", [])

if not st.session_state.thread_id:
    restored = st.query_params.get("thread")
    if restored:
        st.session_state.thread_id = restored
        st.session_state.load_thread = True


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR review agent")
st.caption(
    f"Auto-approve ≥ {AUTO_APPROVE_THRESHOLD:.0%} · "
    f"Human approval {ESCALATE_THRESHOLD:.0%}–{AUTO_APPROVE_THRESHOLD:.0%} · "
    f"Escalate < {ESCALATE_THRESHOLD:.0%}"
)


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    try:
        recent = asyncio.run(_recent_threads())
    except Exception as exc:
        recent = []
        st.caption(f"Audit log unavailable: {exc}")
    st.session_state.recent_rows = recent
    if not recent:
        st.caption("No sessions yet. Run a review to record one.")
    else:
        st.dataframe(
            [
                {
                    "thread_id": row["thread_id"],
                    "pr_url": row["pr_url"],
                    "last_event": str(row["last_event"]),
                    "worst_risk": row["worst_risk"],
                    "events": row["events"],
                    "open": ":material/play_arrow: Open",
                }
                for row in recent
            ],
            column_config={
                "thread_id": st.column_config.TextColumn("Thread"),
                "pr_url": st.column_config.LinkColumn("Pull request"),
                "last_event": st.column_config.TextColumn("Last event"),
                "worst_risk": st.column_config.TextColumn("Worst risk"),
                "events": st.column_config.NumberColumn("Events"),
                "open": st.column_config.ButtonColumn(
                    "Open",
                    on_click=_open_recent_session,
                    key="session_open",
                ),
            },
            hide_index=True,
            alt="Recent review sessions from the audit log",
        )


# Load after the sidebar so an "Open" click in this run is visible, and
# before the PR URL widget so the restored URL can be written into it.
if st.session_state.load_thread and st.session_state.thread_id:
    st.session_state.load_thread = False
    with st.spinner("Loading session..."):
        try:
            values, payload = asyncio.run(_load_checkpoint(st.session_state.thread_id))
        except Exception as exc:
            st.error(str(exc), title="Could not load session")
            values, payload = {}, None
    pr_url_from_state = (values or {}).get("pr_url") or st.session_state.pr_url
    if pr_url_from_state:
        st.session_state.pr_url = pr_url_from_state
        st.session_state.pr_url_input = pr_url_from_state
    if payload is not None:
        st.session_state.interrupt_payload = payload
        st.session_state.final = None
    elif values:
        st.session_state.interrupt_payload = None
        st.session_state.final = values


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL",
        key="pr_url_input",
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", width="stretch")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    with st.container(border=True):
        st.subheader(f"Approval requested — confidence {conf:.0%}")
        st.caption(payload["confidence_reasoning"])
        st.markdown(payload["summary"])

        for comment in payload.get("comments", []):
            st.markdown(
                f"- **[{comment['severity']}]** "
                f"`{comment['file']}:{comment.get('line') or '?'}` — {comment['body']}"
            )

        with st.expander("Diff"):
            st.code(payload.get("diff_preview", ""), language="diff")

        feedback = st.text_input("Feedback (optional)", key="approval_feedback")
        approve, reject, edit = st.columns(3)
        if approve.button("Approve", type="primary", width="stretch", icon=":material/check:"):
            return {"choice": "approve", "feedback": feedback}
        if reject.button("Reject", width="stretch", icon=":material/close:"):
            return {"choice": "reject", "feedback": feedback}
        if edit.button("Edit", width="stretch", icon=":material/edit:"):
            return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    with st.container(border=True):
        st.subheader(f"Strong escalation — confidence {conf:.0%}")
        st.caption(payload["confidence_reasoning"])
        if payload.get("risk_factors"):
            st.error(", ".join(payload["risk_factors"]), title="Risk factors")
        st.markdown(payload["summary"])

        questions = payload.get("questions") or []
        with st.form("escalation"):
            answers: dict[str, str] = {}
            for index, question in enumerate(questions):
                answers[question] = st.text_input(question, key=f"esc_q_{index}")
            submitted_answers = st.form_submit_button("Submit answers", width="stretch")
        if submitted_answers:
            return answers
    return None


def _render_final(final: dict) -> None:
    action = final.get("final_action") or "?"
    analysis = final.get("analysis")
    summary = _analysis_field(analysis, "summary") or ""
    confidence = _analysis_field(analysis, "confidence")
    pr = st.session_state.pr_url

    if "commit_failed" in str(action):
        st.error(
            "The agent decided to post a comment, but the GitHub call failed. "
            "Check GITHUB_TOKEN and try the same thread again.",
            title=f"Post failed · {action}",
        )
    elif str(action).startswith("auto") or str(action).startswith("committed"):
        st.success(
            summary or "The agent posted the review comment.",
            title=f"Review finished · {action}",
        )
    elif action == "rejected":
        st.warning("No comment was posted.", title="Rejected")
    else:
        st.info(str(action), title="Final action")

    if isinstance(confidence, (int, float)):
        st.metric("Confidence", f"{confidence:.0%}")
    if summary and "commit_failed" in str(action):
        st.markdown(summary)
    if pr and "commit_failed" not in str(action) and action != "rejected":
        st.link_button(
            "View comment on GitHub",
            pr,
            icon=":material/open_in_new:",
        )
    st.caption(
        f"thread_id = {st.session_state.thread_id}  ·  replay: "
        f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = _new_thread_id()
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.load_thread = False

    with st.spinner("Fetching PR and asking the LLM..."):
        try:
            result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))
        except Exception as exc:
            st.error(str(exc), title="Review failed")
            st.stop()

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload.get("kind")
    if kind == "approval_request":
        answer = render_approval_card(payload)
    elif kind == "escalation":
        answer = render_escalation_card(payload)
    else:
        st.error(f"Unknown interrupt kind: {kind}", title="Cannot resume")
        answer = None
    if answer is not None:
        with st.spinner("Resuming..."):
            try:
                result = asyncio.run(run_graph(
                    st.session_state.pr_url,
                    st.session_state.thread_id,
                    resume_value=answer,
                ))
            except Exception as exc:
                st.error(str(exc), title="Resume failed")
                st.stop()
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

if st.session_state.final is not None:
    _render_final(st.session_state.final)

if st.session_state.thread_id and st.query_params.get("thread") != st.session_state.thread_id:
    st.query_params["thread"] = st.session_state.thread_id
