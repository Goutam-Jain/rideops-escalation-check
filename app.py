"""RideOps Escalation Readiness Check (v0) - Streamlit app for L1 support agents.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rideops import metrics as M  # noqa: E402
from rideops.checklists import checklist_for  # noqa: E402
from rideops.data import DataValidationError, load_tickets  # noqa: E402
from rideops.escalation import (  # noqa: E402
    build_escalation_note, evaluate_readiness, find_related_eng_tickets, log_check, read_log,
)

DEFAULT_DATA = ROOT / "data" / "rideops_support_tickets.csv"
LOG_PATH = ROOT / "data" / "escalation_checks.csv"
PRIORITY_ORDER = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
BLUE, ORANGE, AQUA, GRAY, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#52514e"

st.set_page_config(page_title="RideOps · Escalation Check", page_icon="🛠️", layout="wide")


# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading tickets…")
def load(data_bytes: bytes):
    return load_tickets(io.BytesIO(data_bytes))


def get_data():
    st.sidebar.markdown("### Data")
    up = st.sidebar.file_uploader("Use a newer ticket export (same columns)", type=["csv"])
    if up is not None:
        raw, label = up.getvalue(), up.name
    elif DEFAULT_DATA.exists():
        raw, label = DEFAULT_DATA.read_bytes(), DEFAULT_DATA.name
    else:
        st.error(f"No data file found at {DEFAULT_DATA}. Upload a CSV in the sidebar.")
        st.stop()
    try:
        df, report = load(raw)
    except DataValidationError as exc:
        st.error(f"This file can't be used: {exc}")
        st.stop()
    st.sidebar.caption(f"{label} · {len(df):,} tickets · data as of {df.attrs['snapshot'] - pd.Timedelta(minutes=1):%d %b %Y}")
    if report.warnings:
        with st.sidebar.expander(f"⚠️ {len(report.warnings)} data-quality note(s)"):
            for w in report.warnings:
                st.write("•", w)
    return df


def fmt_days(hours: float) -> str:
    return "–" if pd.isna(hours) else (f"{hours:.0f} h" if hours < 48 else f"{hours / 24:.1f} days")


def fmt_pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.0%}"


# --------------------------------------------------------------------------- tab 1
def queue_for(df: pd.DataFrame, agent: str, show_all: bool) -> pd.DataFrame:
    q = df[df["status"].eq("Open - with L1")]
    if not show_all:
        q = q[q["assigned_agent"] == agent]
    q = q.assign(
        returned_by_eng=q["is_bounce"],
        _p=q["priority"].map(PRIORITY_ORDER),
    ).sort_values(["returned_by_eng", "_p", "age_h"], ascending=[False, True, False])
    return q


def history_panel(df: pd.DataFrame, category: str, description: str) -> None:
    prof = M.category_escalation_profile(df, category, description)
    scope = "tickets with this exact summary" if prof["scope"] == "description" else f"'{category}' tickets"
    st.markdown(f"##### What happened to past escalations of {scope}")
    st.caption(f"Based on {prof['escalations']} escalations out of {prof['tickets']} tickets in the data.")
    c1, c2 = st.columns(2)
    c1.metric("Fixed by engineering", fmt_pct(prof["fix_rate"]))
    c2.metric("Sent back to L1", fmt_pct(prof["bounce_rate"]),
              help="Engineering returned it as 'Insufficient info' or 'Config issue'. Every one of these needed at least 2 handoffs.")
    c3, c4 = st.columns(2)
    c3.metric("Median close time, escalated", fmt_days(prof["median_ttr_escalated_h"]),
              help=f"vs {fmt_days(prof['median_ttr_l1_h'])} when resolved at L1 without escalating")
    c4.metric("…if sent back first", fmt_days(prof["median_ttr_bounced_h"]),
              help="Median time to final close for escalations that engineering returned to L1")
    if prof["outcomes"]:
        o = pd.Series(prof["outcomes"]).sort_values()
        colors = [ORANGE if "returned to L1" in k else (BLUE if k == "Fixed" else GRAY) for k in o.index]
        fig = go.Figure(go.Bar(x=o.values, y=o.index, orientation="h", marker_color=colors,
                               hovertemplate="%{y}: %{x} escalations<extra></extra>", text=o.values, textposition="outside"))
        fig.update_layout(height=200, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(visible=False),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(size=12))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if prof["insufficient_info_rate"] + prof["config_rate"] > 0:
        st.info(f"**{fmt_pct(prof['insufficient_info_rate'])}** of these escalations came back for missing information and "
                f"**{fmt_pct(prof['config_rate'])}** turned out to be a configuration issue L1 could check. The checklist below targets both.")


def related_panel(df: pd.DataFrame, category: str, description: str, as_of, exclude: str | None) -> pd.DataFrame:
    rel = find_related_eng_tickets(df, category, description, as_of=as_of, exclude_ticket=exclude)
    st.markdown("##### Recent similar escalations (last 14 days)")
    if rel.empty:
        st.caption("No similar escalation in this category in the last 14 days.")
        return rel
    st.caption("Context only: check whether one of these is the same problem before asking engineering to start over. "
               "A text match does not prove it is the same issue.")
    show = rel.assign(
        raised=rel["created_at"].dt.strftime("%d %b %H:%M"),
        state=rel["open_then"].map({True: "still open", False: "closed"}),
        outcome=rel["eng_outcome"].fillna("in progress"),
        match=rel["similarity"].map(lambda s: f"{s:.0%}"),
    )[["eng_ticket_id", "raised", "state", "outcome", "match", "description"]]
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"description": st.column_config.TextColumn(width="large")})
    return rel


def checklist_form(ticket: dict, category: str, agent: str, related: pd.DataFrame) -> None:
    spec = checklist_for(category)
    tid = ticket["ticket_id"]
    st.markdown("##### Escalation checklist")
    st.caption("Draft v0 checklist. To be reviewed with Engineering in week 1. Every field ends up in the note engineering receives.")
    if spec["route"]:
        st.warning(spec["route"])
    with st.form(key=f"form_{tid}"):
        answers, checks = {}, {}
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Information to collect** (required)")
            for key, label in spec["info"].items():
                answers[key] = st.text_input(label, key=f"{tid}_i_{key}")
        with right:
            st.markdown("**Checks done at L1** (required)")
            for key, label in spec["checks"].items():
                checks[key] = st.checkbox(label, key=f"{tid}_c_{key}")
        submitted = st.form_submit_button("Check readiness", type="primary")
    if submitted:
        r = evaluate_readiness(category, answers, checks, related)
        note = build_escalation_note(ticket, category, answers, checks, r, related, agent)
        st.session_state[f"result_{tid}"] = (r, note)
        log_check(LOG_PATH, tid, agent, category, ticket.get("priority", ""), r, related, action="checked")

    if f"result_{tid}" in st.session_state:
        r, note = st.session_state[f"result_{tid}"]
        st.progress(r.completeness, text=f"Checklist {r.completeness:.0%} complete")
        if r.verdict == "READY":
            st.success("✅ Ready to escalate. Paste the note below into the engineering ticket.")
        elif r.verdict == "REVIEW_ROUTE":
            st.warning("🔀 Complete, but this category usually isn't an engineering issue. Confirm before escalating.")
        else:
            st.error(f"⛔ Not ready: {len(r.missing_info)} item(s) of information and {len(r.missing_checks)} check(s) missing.")
            for m in r.missing_info:
                st.write("• Collect:", m)
            for m in r.missing_checks:
                st.write("• Do / confirm:", m)
        for n in r.notes:
            st.info(n)
        with st.expander("Escalation note (preview)", expanded=r.verdict != "NOT_READY"):
            st.markdown(note)
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download note (.md)", note, file_name=f"{tid}_escalation_note.md", disabled=r.verdict == "NOT_READY")
        if c2.button("Mark as escalated", disabled=r.verdict == "NOT_READY", key=f"esc_{tid}"):
            log_check(LOG_PATH, tid, agent, category, ticket.get("priority", ""), r, related, action="escalated")
            st.toast("Logged as escalated")
        if c3.button("Resolving at L1 instead", key=f"l1_{tid}"):
            log_check(LOG_PATH, tid, agent, category, ticket.get("priority", ""), r, related, action="resolved_at_l1")
            st.toast("Logged as resolved at L1")


def tab_escalate(df: pd.DataFrame, agent: str) -> None:
    snap = df.attrs["snapshot"]
    mode = st.radio("Ticket", ["From my open queue", "New ticket (not in the export)"], horizontal=True, label_visibility="collapsed")
    if mode == "From my open queue":
        show_all = st.toggle("Show every agent's open L1 tickets", value=False)
        q = queue_for(df, agent, show_all)
        if q.empty:
            st.info("No open L1 tickets for you in this export. Toggle to see the whole team, or try a new ticket.")
            return
        n_ret = int(q["returned_by_eng"].sum())
        st.markdown(f"**{len(q)} open ticket(s) with L1**" + (f" · **{n_ret} sent back by engineering**, listed first" if n_ret else ""))
        view = q.assign(
            age=q["age_h"].map(fmt_days),
            sent_back=q["eng_outcome"].where(q["returned_by_eng"], ""),
        )[["ticket_id", "priority", "age", "sent_back", "category", "description", "client_name"] + (["assigned_agent"] if show_all else [])]
        st.dataframe(view, hide_index=True, width="stretch", height=min(35 * (len(view) + 1) + 3, 320))
        labels = {r.ticket_id: f"{r.ticket_id} · {r.priority} · {r.category} · {r.description[:50]}" for r in q.itertuples()}
        tid = st.selectbox("Open a ticket", list(labels), format_func=labels.get)
        row = q[q["ticket_id"] == tid].iloc[0]
        ticket = row.to_dict()
        category, description = row["category"], row["description"]
        with st.container(border=True):
            st.markdown(f"### {tid} · {row['priority']} · {category}")
            st.write(f"**{description}** · {row['client_name']} ({row['client_tier']}) · via {row['channel']} · open {fmt_days(row['age_h'])}")
            if row["is_bounce"]:
                st.error(f"Engineering sent this back ({row['eng_outcome']}) after {row['eng_handoffs']} handoff(s), "
                         f"ticket {row['eng_ticket_id']}. Fill the gaps before re-escalating.")
            elif row["is_escalated"]:
                st.caption(f"Previously escalated: {row['eng_ticket_id']} · last outcome: {row['eng_outcome']}")
        exclude = tid
    else:
        c1, c2, c3 = st.columns([2, 1, 2])
        category = c1.selectbox("Category", sorted(df["category"].unique()))
        priority = c2.selectbox("Priority", ["P1", "P2", "P3", "P4"], index=2)
        client = c3.selectbox("Client", sorted(df["client_name"].unique()))
        description = st.text_input("One-line summary of the issue", placeholder="e.g. Cancelled bookings still appearing in tomorrow's roster")
        if len(description.strip()) < 8:
            st.info("Enter a one-line summary (at least 8 characters) to see history and the checklist.")
            return
        tier = df.loc[df["client_name"] == client, "client_tier"].iloc[0]
        ticket = {"ticket_id": "NEW", "priority": priority, "client_name": client, "client_tier": tier, "description": description.strip()}
        exclude = None

    left, right = st.columns([1, 1])
    with left:
        history_panel(df, category, description)
    with right:
        related = related_panel(df, category, description, as_of=snap, exclude=exclude)
    st.divider()
    checklist_form(ticket, category, agent, related)


# --------------------------------------------------------------------------- tab 2
def line(fig: go.Figure, x, y, name: str, color: str, fmt: str) -> None:
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines+markers", line=dict(color=color, width=2),
                             marker=dict(size=7), hovertemplate=f"%{{x|%d %b}}: %{{y:{fmt}}}<extra>{name}</extra>"))


def small_chart(title: str, fig: go.Figure, pct: bool = False, ymin0: bool = True) -> None:
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=260, margin=dict(l=0, r=10, t=40, b=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2), showlegend=len(fig.data) > 1)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", tickformat=".0%" if pct else None, rangemode="tozero" if ymin0 else None)
    fig.update_xaxes(showgrid=False, tickformat="%d %b")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def tab_kpis(df: pd.DataFrame, agent: str) -> None:
    st.markdown("Weekly escalation-quality measures. The **primary** measure this tool should move is the share of escalations sent back to L1. "
                "The **guardrails** catch work being pushed around instead of removed.")
    c1, c2 = st.columns(2)
    who = c1.selectbox("Agent", ["Whole team", agent] + sorted(a for a in df["assigned_agent"].unique() if a != agent))
    cats = c2.multiselect("Categories", sorted(df["category"].unique()), placeholder="All categories")
    d = df if who == "Whole team" else df[df["assigned_agent"] == who]
    if cats:
        d = d[d["category"].isin(cats)]
    if d.empty:
        st.info("No tickets for this selection.")
        return
    d.attrs.update(df.attrs)
    wk = M.weekly_kpis(d)
    first_full = wk["week_start"].min() + pd.Timedelta(days=7)
    last_full = df.attrs["snapshot"] - pd.Timedelta(days=7)
    wk = wk[(wk["week_start"] >= first_full) & (wk["week_start"] <= last_full - pd.Timedelta(days=0))]
    hk = M.headline_kpis(d)
    k = st.columns(5)
    k[0].metric("Sent back to L1 (of escalations)", fmt_pct(hk["bounce_rate"]), help="Primary measure. 'Insufficient info' + 'Config issue' outcomes ÷ escalations.")
    k[1].metric("Eng handoffs per escalation", f"{hk['avg_handoffs_per_escalation']:.2f}" if pd.notna(hk["avg_handoffs_per_escalation"]) else "–")
    k[2].metric("Escalation rate", fmt_pct(hk["escalation_rate"]), help="Guardrail: should fall a little, not crash.")
    k[3].metric("Reopen rate", fmt_pct(hk["reopen_rate"]), help="Guardrail: must not rise as escalations fall.")
    k[4].metric("Open now (L1 / eng)", f"{hk['open_with_l1']} / {hk['open_with_eng']}")
    st.caption("Whole-period values for the selection. Charts show full weeks only (6 Jul – 21 Sep). Escalation cohorts from the last 2 weeks are provisional: "
               "some are still open and may yet bounce.")

    a, b = st.columns(2)
    with a:
        fig = go.Figure(); line(fig, wk["week_start"], wk["bounce_rate"], "Sent back to L1", ORANGE, ".0%")
        small_chart("Primary · Escalations sent back to L1", fig, pct=True)
    with b:
        fig = go.Figure(); line(fig, wk["week_start"], wk["avg_handoffs_per_escalation"], "Handoffs per escalation", BLUE, ".2f")
        small_chart("Engineering handoffs per escalation", fig)
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        line(fig, wk["week_start"], wk["escalation_rate"], "Escalation rate", BLUE, ".0%")
        line(fig, wk["week_start"], wk["reopen_rate"], "Reopen rate", ORANGE, ".0%")
        small_chart("Guardrails · escalation rate and reopen rate", fig, pct=True)
    with b:
        fig = go.Figure()
        line(fig, wk["week_start"], wk["created"], "Created", BLUE, ",.0f")
        line(fig, wk["week_start"], wk["closed"], "Closed", ORANGE, ",.0f")
        small_chart("Outcome · tickets created vs closed per week", fig)
    fig = go.Figure(); line(fig, wk["week_start"] + pd.Timedelta(days=6), wk["open_at_week_end"], "Open tickets", BLUE, ",.0f")
    small_chart("Outcome · open tickets at week end (team-wide when no filter)", fig)

    with st.expander("Weekly table"):
        st.dataframe(wk.round(3), hide_index=True, width="stretch")

    st.markdown("#### Pilot log from this tool")
    log = read_log(LOG_PATH)
    if log.empty:
        st.caption("No readiness checks logged yet. Each 'Check readiness' click and each escalate / resolve-at-L1 decision adds a row to "
                   f"`{LOG_PATH.relative_to(ROOT)}`. In a pilot this gives the leading measure: % of escalations sent with a complete note.")
    else:
        esc_log = log[log["action"] == "escalated"]
        c = st.columns(3)
        c[0].metric("Checks run", len(log[log["action"] == "checked"]))
        c[1].metric("Escalations sent through the tool", len(esc_log))
        c[2].metric("…with a complete checklist", fmt_pct((esc_log["completeness"] >= 1).mean()) if len(esc_log) else "–")
        st.dataframe(log.tail(50).iloc[::-1], hide_index=True, width="stretch")


# --------------------------------------------------------------------------- tab 3
def tab_about(df: pd.DataFrame) -> None:
    hk = M.headline_kpis(df)
    esc = df[df["is_escalated"]]
    seg = M.backlog_hours_by(df, "escalation_segment")
    b = seg.loc["Bounced back to L1"] if "Bounced back to L1" in seg.index else None
    st.markdown(f"""
#### Who this is for
L1 support agents, at the moment they are about to escalate a ticket to engineering.

#### The problem it targets (computed live from the loaded data)
- {fmt_pct(hk['escalation_rate'])} of tickets are escalated. Engineering fixes {fmt_pct(hk['eng_fix_rate'])} of them.
- **{fmt_pct(hk['bounce_rate'])} ({int(esc['is_bounce'].sum())} tickets) are sent back to L1** as *Insufficient info* or *Config issue*. Each one needs another round trip.
""" + (f"- Those tickets are {b['ticket_share']:.0%} of all tickets but **{b['backlog_share']:.0%} of the time tickets spend open**, the biggest single block of backlog." if b is not None else "") + """

#### What the tool does
1. **Queue:** your open L1 tickets, with the ones engineering sent back listed first.
2. **History:** what happened to past escalations in the same category (or the same summary, if it has at least 10 escalations).
3. **Recent similar escalations:** escalated tickets in the same category from the last 14 days with a similar summary (TF-IDF cosine ≥ 0.6), and whether they are still open.
4. **Checklist → verdict:** required information and L1 checks per category. Rule: anything missing → *Not ready*. Complete → *Ready* (or *Check routing* for categories that usually aren't engineering issues).
5. **Escalation note:** a structured note to paste into the engineering ticket. Every check is logged, so the pilot can be measured.

#### What it deliberately does not do
- **No prediction of whether an escalation will bounce.** I tested it: a model on category, priority, channel, tier, agent and summary text scored AUC ≈ 0.5 on a time-split test, no better than chance. The data does not record *what* was in each escalation, which is the thing that matters. So the rules are transparent instead.
- **No automatic "known issue" blocking.** In a backtest, a similar-summary match did not separate true duplicates from other escalations (precision ≈ base rate). It is shown as context only.
- **The checklists are a draft.** The data shows *that* escalations come back for missing information or config. It does not show *which* information. The lists must be reviewed with engineering before a pilot.

#### Data rules
- Blank `eng_outcome` / `eng_ticket_id` = never escalated. Blank `final_closed_at` = still open. Blank `csat_score` = not rated (not zero).
- "Open at time t" = created on or before t and not yet finally closed. Reopen times are not recorded, so this is a slight upper bound.
- Open tickets are aged to the end of the data-pull day (30 Sep 2026).
- This is historical data. The tool has **not** been shown to reduce the backlog. That needs a pilot (see the write-up).
""")


# --------------------------------------------------------------------------- main
def main() -> None:
    df = get_data()
    st.sidebar.markdown("### You")
    agents = sorted(df["assigned_agent"].unique())
    q_counts = df[df["status"] == "Open - with L1"]["assigned_agent"].value_counts()
    agent = st.sidebar.selectbox("Signed in as (L1 agent)", agents,
                                 index=agents.index(q_counts.idxmax()) if not q_counts.empty else 0)
    st.title("Escalation Readiness Check")
    st.caption("For L1 agents: before a ticket goes to engineering, make sure it won't come straight back.")
    t1, t2, t3 = st.tabs(["Escalate a ticket", "Escalation quality (weekly)", "How this works"])
    with t1:
        tab_escalate(df, agent)
    with t2:
        tab_kpis(df, agent)
    with t3:
        tab_about(df)


if __name__ == "__main__":
    main()
