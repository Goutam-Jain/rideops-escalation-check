"""KPI and diagnostic calculations. Pure functions over the enriched ticket frame."""
from __future__ import annotations

import numpy as np
import pandas as pd


def open_count_at(df: pd.DataFrame, when: pd.Timestamp) -> int:
    """Tickets open at `when`: created on/before it and not yet finally closed.

    Limitation: reopen timestamps are not in the data, so a ticket that was
    closed and later reopened is counted as open for the whole gap. This makes
    the series a slight upper bound; the error is small (10% of tickets reopen).
    """
    created = df["created_at"] <= when
    not_closed = df["final_closed_at"].isna() | (df["final_closed_at"] > when)
    return int((created & not_closed).sum())


def backlog_series(df: pd.DataFrame, freq: str = "W-SUN") -> pd.DataFrame:
    """Open-ticket count at the end of each period, plus the L1 / engineering split
    for the final snapshot (status is only known as of the data pull)."""
    start = df["created_at"].min().normalize()
    snap = df.attrs.get("snapshot", df["created_at"].max())
    ends = pd.date_range(start, snap, freq=freq)
    ends = [e + pd.Timedelta(hours=23, minutes=59) for e in ends]
    if not ends or ends[-1] < snap - pd.Timedelta(minutes=1):
        ends.append(snap - pd.Timedelta(minutes=1))
    return pd.DataFrame({"period_end": ends, "open_tickets": [open_count_at(df, e) for e in ends]})


def weekly_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ISO week (Mon-Sun) of ticket creation.

    Escalation-quality metrics are cohort metrics: a week's bounce rate is about
    the escalations *created* that week. Recent cohorts are not final yet
    (open tickets may still bounce) - the `cohort_open` column shows how many.
    """
    g = df.groupby("created_week")
    esc = df[df["is_escalated"]].groupby("created_week")
    closed = df.dropna(subset=["final_closed_at"])
    closed_wk = closed.groupby(closed["final_closed_at"].dt.to_period("W-SUN").dt.start_time).size()

    out = pd.DataFrame({
        "created": g.size(),
        "escalation_rate": g["is_escalated"].mean(),
        "escalations": esc.size(),
        "bounce_rate": esc["is_bounce"].mean(),
        "avg_handoffs_per_escalation": esc["eng_handoffs"].mean(),
        "reopen_rate": g["is_reopened"].mean(),
        "median_frt_h": g["frt_h"].median(),
        "median_ttr_h": g["ttr_h"].median(),
        "csat_mean": g["csat_score"].mean(),
        "csat_responses": g["csat_score"].count(),
        "cohort_open": g["is_open"].sum(),
    })
    out["closed"] = closed_wk.reindex(out.index).fillna(0).astype(int)
    out["net_flow"] = out["closed"] - out["created"]
    bl = backlog_series(df)
    bl["week"] = bl["period_end"].dt.to_period("W-SUN").dt.start_time
    out["open_at_week_end"] = bl.groupby("week")["open_tickets"].last().reindex(out.index)
    out.index.name = "week_start"
    return out.reset_index()


def headline_kpis(df: pd.DataFrame) -> dict:
    """Baseline numbers for the KPI table. Everything is computed, nothing typed in."""
    esc = df[df["is_escalated"]]
    return {
        "tickets": len(df),
        "open_now": int(df["is_open"].sum()),
        "open_with_l1": int(df["status"].eq("Open - with L1").sum()),
        "open_with_eng": int(df["status"].eq("Open - with Engineering").sum()),
        "escalation_rate": float(df["is_escalated"].mean()),
        "bounce_rate": float(esc["is_bounce"].mean()) if len(esc) else np.nan,
        "eng_fix_rate": float(esc["is_fixed"].mean()) if len(esc) else np.nan,
        "avg_handoffs_per_escalation": float(esc["eng_handoffs"].mean()) if len(esc) else np.nan,
        "reopen_rate": float(df["is_reopened"].mean()),
        "median_frt_h": float(df["frt_h"].median()),
        "median_ttr_h": float(df["ttr_h"].median()),
        "median_ttr_escalated_h": float(esc["ttr_h"].median()) if len(esc) else np.nan,
        "median_ttr_l1_h": float(df.loc[~df["is_escalated"], "ttr_h"].median()),
        "csat_mean": float(df["csat_score"].mean()),
        "csat_response_rate": float(df["csat_score"].notna().mean()),
    }


def backlog_hours_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Share of total ticket-hours in the system, by a grouping column.

    By Little's law, average backlog = sum of time-in-system / length of period,
    so a group's share of ticket-hours is its share of the average open backlog.
    """
    t = df.groupby(by).agg(tickets=("ticket_id", "size"), ticket_hours=("time_in_system_h", "sum"),
                           median_h=("time_in_system_h", "median"))
    t["ticket_share"] = t["tickets"] / t["tickets"].sum()
    t["backlog_share"] = t["ticket_hours"] / t["ticket_hours"].sum()
    return t.sort_values("ticket_hours", ascending=False)


def littles_law_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Average open count per month (hourly samples) vs arrival rate.

    implied_days_in_system = avg_open / arrivals_per_day. If backlog grows faster
    than arrivals, tickets are also staying longer.
    """
    snap = df.attrs.get("snapshot", df["created_at"].max())
    hours = pd.date_range(df["created_at"].min().ceil("h"), snap - pd.Timedelta(hours=1), freq="h")
    created = df["created_at"].to_numpy()
    end = df["final_closed_at"].fillna(snap).to_numpy()
    counts = [int(((created <= h) & (end > h)).sum()) for h in hours.to_numpy()]
    s = pd.Series(counts, index=hours)
    month = s.index.to_period("M").astype(str)
    avg_open = s.groupby(month).mean()
    days_obs = s.groupby(month).size() / 24.0
    arrivals = df.groupby("created_month").size().reindex(avg_open.index)
    out = pd.DataFrame({"avg_open": avg_open, "arrivals": arrivals, "days_observed": days_obs})
    out["arrivals_per_day"] = out["arrivals"] / out["days_observed"]
    out["implied_days_in_system"] = out["avg_open"] / out["arrivals_per_day"]
    return out


def agent_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """Per-agent metrics with an expected value from the agent's own ticket mix.

    expected_* = mean of the team rate for each (category, priority) cell the
    agent handled, so an agent who gets harder tickets is not penalised.
    """
    d = df.copy()
    cell = ["category", "priority"]
    d["exp_esc"] = d.groupby(cell)["is_escalated"].transform("mean")
    d["exp_reopen"] = d.groupby(cell)["is_reopened"].transform("mean")
    d["exp_bounce"] = d.groupby(cell)["is_bounce"].transform("mean")
    g = d.groupby("assigned_agent")
    out = pd.DataFrame({
        "tickets": g.size(),
        "median_ttr_h": g["ttr_h"].median(),
        "escalation_rate": g["is_escalated"].mean(),
        "expected_escalation_rate": g["exp_esc"].mean(),
        "bounces": g["is_bounce"].sum(),
        "expected_bounces": g["exp_bounce"].sum(),
        "reopen_rate": g["is_reopened"].mean(),
        "expected_reopen_rate": g["exp_reopen"].mean(),
        "csat_mean": g["csat_score"].mean(),
        "open_now": g["is_open"].sum(),
    })
    out["bounce_rate_of_escalations"] = d[d["is_escalated"]].groupby("assigned_agent")["is_bounce"].mean()
    return out.sort_values("tickets", ascending=False)


def category_escalation_profile(df: pd.DataFrame, category: str | None = None,
                                description: str | None = None, min_n: int = 10) -> dict:
    """Historical escalation evidence for a category (and description, if enough history).

    Falls back from description -> category when the description has fewer than
    `min_n` escalations, so the agent never sees a rate built on 2 tickets.
    """
    scope = "category"
    base = df[df["category"] == category] if category else df
    if description:
        d = base[base["description"] == description]
        if int(d["is_escalated"].sum()) >= min_n:
            base, scope = d, "description"
    esc = base[base["is_escalated"]]
    outcomes = esc["eng_outcome"].value_counts()
    return {
        "scope": scope,
        "tickets": int(len(base)),
        "escalations": int(len(esc)),
        "escalation_rate": float(base["is_escalated"].mean()) if len(base) else np.nan,
        "bounce_rate": float(esc["is_bounce"].mean()) if len(esc) else np.nan,
        "fix_rate": float(esc["is_fixed"].mean()) if len(esc) else np.nan,
        "insufficient_info_rate": float((esc["eng_outcome"] == "Insufficient info - returned to L1").mean()) if len(esc) else np.nan,
        "config_rate": float((esc["eng_outcome"] == "Config issue - returned to L1").mean()) if len(esc) else np.nan,
        "duplicate_rate": float((esc["eng_outcome"] == "Duplicate of known issue").mean()) if len(esc) else np.nan,
        "avg_handoffs": float(esc["eng_handoffs"].mean()) if len(esc) else np.nan,
        "median_ttr_escalated_h": float(esc["ttr_h"].median()) if len(esc) else np.nan,
        "median_ttr_l1_h": float(base.loc[~base["is_escalated"], "ttr_h"].median()) if len(base) else np.nan,
        "median_ttr_bounced_h": float(esc.loc[esc["is_bounce"], "ttr_h"].median()) if esc["is_bounce"].any() else np.nan,
        "csat_bounced": float(esc.loc[esc["is_bounce"], "csat_score"].mean()) if esc["is_bounce"].any() else np.nan,
        "csat_l1": float(base.loc[~base["is_escalated"], "csat_score"].mean()) if len(base) else np.nan,
        "outcomes": outcomes.to_dict(),
    }
