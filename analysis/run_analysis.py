"""Reproducible diagnosis: every number in the write-up comes from this script.

Run:  python analysis/run_analysis.py
Writes: docs/analysis_results.md, docs/metrics.json, docs/figures/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rideops import metrics as M  # noqa: E402
from rideops.data import load_tickets  # noqa: E402
from rideops.escalation import find_related_eng_tickets  # noqa: E402

DATA = ROOT / "data" / "rideops_support_tickets.csv"
DOCS = ROOT / "docs"
FIG = DOCS / "figures"

# Reference palette (dataviz skill, light mode)
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

out: list[str] = []
J: dict = {}


def h(title: str) -> None:
    out.append(f"\n## {title}\n")


def p(text: str = "") -> None:
    out.append(text)


def table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> None:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    out.append(d.to_markdown())
    out.append("")


def pct(x: float) -> str:
    return f"{x:.1%}"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    df, report = load_tickets(DATA)
    snap = df.attrs["snapshot"]
    esc = df[df["is_escalated"]]

    # ------------------------------------------------------------------ 1. data quality
    h("1. Data quality")
    p(f"Rows: {len(df)}, unique ticket_ids: {df['ticket_id'].nunique()}, fully duplicated rows: {int(df.duplicated(subset=[c for c in df.columns if c not in ('created_week',)]).sum())}")
    p(f"Created range: {df['created_at'].min()} to {df['created_at'].max()}; snapshot used for ageing: {snap}")
    p("Blank counts (blanks are meaningful, see data.py docstring):")
    table(df[["first_response_at", "eng_ticket_id", "eng_outcome", "first_closed_at", "final_closed_at", "csat_score"]].isna().sum().to_frame("blank"))
    p("Consistency warnings: " + ("; ".join(report.warnings) if report.warnings else "none"))
    p(f"Distinct description texts: {df['description'].nunique()} across {len(df)} tickets (templated summaries; free text is thin).")
    J["data_quality_warnings"] = report.warnings

    # ------------------------------------------------------------------ 2. volume
    h("2. Volume")
    monthly = df.groupby("created_month").size()
    table(monthly.to_frame("tickets"))
    growth = monthly.iloc[-1] / monthly.iloc[0] - 1
    p(f"Jul -> Sep ticket growth: {pct(growth)} (July starts 2 Jul, so July is 30 days like September).")
    cat_month = pd.crosstab(df["category"], df["created_month"])
    cat_month["growth_jul_sep"] = cat_month.iloc[:, -1] / cat_month.iloc[:, 0] - 1
    cat_month["added_tickets"] = cat_month.iloc[:, 2] - cat_month.iloc[:, 0]
    table(cat_month.sort_values("added_tickets", ascending=False))
    p(f"Active clients per month: {df.groupby('created_month')['client_id'].nunique().to_dict()}")
    per_agent = df.groupby("created_month").size() / df["assigned_agent"].nunique()
    p(f"Tickets per agent per month (8 agents throughout): {per_agent.round(1).to_dict()}")
    for c in ["priority", "client_tier", "channel"]:
        table(pd.crosstab(df[c], df["created_month"], margins=True))
    J["monthly_tickets"] = monthly.to_dict()
    J["growth_jul_sep"] = growth

    # ------------------------------------------------------------------ 3. backlog
    h("3. Backlog")
    month_ends = {m: pd.Timestamp(m) for m in ["2026-07-31 23:59", "2026-08-31 23:59", "2026-09-30 23:59"]}
    bl = {k[:7]: M.open_count_at(df, v) for k, v in month_ends.items()}
    p(f"Open tickets at month end (reconstructed from timestamps): {bl}. Change Jul->Sep: {pct(bl['2026-09'] / bl['2026-07'] - 1)}")
    hk = M.headline_kpis(df)
    p(f"Open at snapshot: {hk['open_now']} ({hk['open_with_l1']} with L1, {hk['open_with_eng']} with engineering)")
    o = df[df["is_open"]]
    p(f"Open tickets that have been escalated at least once: {int(o['is_escalated'].sum())} of {len(o)} ({pct(o['is_escalated'].mean())}); "
      f"of those, last outcome was a bounce: {int(o['is_bounce'].sum())}")
    p(f"Open-ticket age (days): median {o['age_h'].median() / 24:.1f}, max {o['age_h'].max() / 24:.1f}; L1 median {o.loc[o['status'] == 'Open - with L1', 'age_h'].median() / 24:.1f}, eng median {o.loc[o['status'] == 'Open - with Engineering', 'age_h'].median() / 24:.1f}")
    table(pd.crosstab(o["category"], o["status"], margins=True).sort_values("All", ascending=False))
    ll = M.littles_law_by_month(df)
    table(ll)
    ratio_open = ll["avg_open"].iloc[-1] / ll["avg_open"].iloc[0]
    ratio_arr = ll["arrivals_per_day"].iloc[-1] / ll["arrivals_per_day"].iloc[0]
    ratio_time = ll["implied_days_in_system"].iloc[-1] / ll["implied_days_in_system"].iloc[0]
    share_arr = np.log(ratio_arr) / np.log(ratio_open)
    p(f"Average open backlog Jul->Sep x{ratio_open:.2f} = arrivals x{ratio_arr:.2f} times time-in-system x{ratio_time:.2f}. "
      f"Log decomposition: {pct(share_arr)} of growth from more arrivals, {pct(1 - share_arr)} from tickets staying longer.")
    J.update(backlog_month_end=bl, open_now=hk["open_now"], open_l1=hk["open_with_l1"], open_eng=hk["open_with_eng"],
             open_escalated=int(o["is_escalated"].sum()), littles=ll.round(3).to_dict(), share_growth_from_arrivals=share_arr)

    # ------------------------------------------------------------------ 4. resolution
    h("4. Resolution performance (hours)")
    table(df[["frt_h", "ttfc_h", "ttr_h"]].describe(percentiles=[.5, .75, .9]).T)
    table(df.groupby("priority")[["frt_h", "ttr_h"]].median())
    by_m = df.groupby(["created_month", "is_escalated"])["ttr_h"].median().unstack()
    by_m.columns = ["resolved_at_L1", "escalated"]
    by_m["median_frt_h"] = df.groupby("created_month")["frt_h"].median()
    by_m["escalation_rate"] = df.groupby("created_month")["is_escalated"].mean()
    table(by_m)
    p("Overall median TTR is flat (~22h) only because the escalation share fell; both segments got slower (mix effect).")
    table(df.groupby("eng_handoffs")["ttr_h"].describe(percentiles=[.5, .9])[["count", "50%", "90%", "mean"]])
    J["ttr_by_handoffs_median"] = df.groupby("eng_handoffs")["ttr_h"].median().to_dict()
    J["ttr_by_month_segment"] = by_m.round(2).to_dict()

    # ------------------------------------------------------------------ 5. escalations
    h("5. Engineering escalations")
    p(f"Escalation rate {pct(hk['escalation_rate'])} ({len(esc)} tickets). Fixed by eng: {pct(hk['eng_fix_rate'])}. "
      f"Bounced back to L1 (config / insufficient info): {pct(hk['bounce_rate'])} ({int(esc['is_bounce'].sum())}). "
      f"Closed without fix (dup / not a bug / not repro): {pct(esc['eng_outcome'].isin(['Duplicate of known issue', 'Not a bug - how-to', 'Not reproducible']).mean())}.")
    table(esc["eng_outcome"].value_counts().to_frame("n").assign(share=lambda x: x["n"] / x["n"].sum()))
    table(pd.crosstab(esc["eng_outcome"], esc["eng_handoffs"]))
    p(f"Total engineering handoffs {int(df['eng_handoffs'].sum())}; handoffs beyond the first: {int(df['extra_handoffs'].sum())} "
      f"({pct(df['extra_handoffs'].sum() / df['eng_handoffs'].sum())}); every bounce has >= 2 handoffs.")
    cat_out = pd.crosstab(esc["category"], esc["eng_outcome"], margins=True)
    table(cat_out.sort_values("All", ascending=False))
    cat_rate = df.groupby("category").agg(tickets=("ticket_id", "size"), escalation_rate=("is_escalated", "mean"))
    cat_rate["bounce_rate_of_esc"] = esc.groupby("category")["is_bounce"].mean()
    table(cat_rate.sort_values("tickets", ascending=False))
    table(pd.crosstab(esc["created_month"], esc["eng_outcome"], normalize="index"))
    bounce_m = esc.groupby("created_month")["is_bounce"].mean()
    insuff_m = esc.groupby("created_month")["eng_outcome"].apply(lambda s: (s == "Insufficient info - returned to L1").mean())
    p(f"Bounce rate by month: {bounce_m.round(3).to_dict()}; insufficient-info share by month: {insuff_m.round(3).to_dict()}")
    table(pd.crosstab(esc["channel"], esc["eng_outcome"], normalize="index"))
    J.update(escalation_rate=hk["escalation_rate"], bounce_rate=hk["bounce_rate"], eng_fix_rate=hk["eng_fix_rate"],
             bounces=int(esc["is_bounce"].sum()), escalations=len(esc), extra_handoffs=int(df["extra_handoffs"].sum()),
             total_handoffs=int(df["eng_handoffs"].sum()), avg_handoffs=hk["avg_handoffs_per_escalation"],
             duplicates=int((esc["eng_outcome"] == "Duplicate of known issue").sum()),
             duplicates_report_export=int(((esc["eng_outcome"] == "Duplicate of known issue") & (esc["category"] == "Report export failing")).sum()),
             bounce_rate_by_month=bounce_m.to_dict())

    # ------------------------------------------------------------------ 6. where the backlog-hours go
    h("6. Pareto: where the backlog-hours go (Little's law)")
    seg = M.backlog_hours_by(df, "escalation_segment")
    table(seg)
    cat = M.backlog_hours_by(df, "category")
    cat["cum_backlog_share"] = cat["backlog_share"].cumsum()
    table(cat)
    base_1 = df.loc[df["eng_handoffs"] == 1, "time_in_system_h"].median()
    multi = df[df["eng_handoffs"] >= 2]
    excess = (multi["time_in_system_h"] - base_1).clip(lower=0).sum()
    p(f"Tickets with >= 2 handoffs: {len(multi)} ({pct(len(multi) / len(df))}). Hours above the 1-handoff median ({base_1:.1f}h): "
      f"{excess:,.0f} = {pct(excess / df['time_in_system_h'].sum())} of all ticket-hours.")
    p(f"Sizing (estimate, not a forecast): halving repeat handoffs would remove ~{pct(excess / 2 / df['time_in_system_h'].sum())} "
      "of average open backlog, if nothing else changed.")
    J.update(segment_backlog=seg[["tickets", "ticket_share", "backlog_share", "median_h"]].round(4).to_dict(),
             category_backlog=cat[["tickets", "ticket_share", "backlog_share"]].round(4).to_dict(),
             multi_handoff_tickets=len(multi), excess_hours_share=excess / df["time_in_system_h"].sum())

    # ------------------------------------------------------------------ 7. agents
    h("7. Agents (mix-adjusted by category x priority)")
    sc = M.agent_scorecard(df)
    table(sc)
    ct_esc = pd.crosstab(df["assigned_agent"], df["is_escalated"])
    ct_reo = pd.crosstab(df["assigned_agent"], df["is_reopened"])
    ct_bnc = pd.crosstab(esc["assigned_agent"], esc["is_bounce"])
    p_esc, p_reo, p_bnc = (chi2_contingency(c)[1] for c in (ct_esc, ct_reo, ct_bnc))
    p(f"Chi-square p-values: escalation rate differs by agent p={p_esc:.1e}; reopen rate p={p_reo:.1e}; "
      f"bounce rate *among escalations* p={p_bnc:.2f} (not significant -> bounces are a team-wide process issue).")
    reopen_share = df[df["is_reopened"]]["assigned_agent"].value_counts(normalize=True)
    p(f"Share of all reopened tickets by agent: {reopen_share.round(3).to_dict()}")
    p(f"Reopen rate excluding the top-reopen agent: {pct(df.loc[df['assigned_agent'] != reopen_share.index[0], 'is_reopened'].mean())}")
    bounce_share = esc[esc["is_bounce"]]["assigned_agent"].value_counts(normalize=True)
    p(f"Share of all bounces by agent: {bounce_share.round(3).to_dict()}")
    table(pd.crosstab(df["assigned_agent"], df["created_month"]))
    J.update(agent_scorecard=sc.round(3).to_dict(), p_agent_bounce=p_bnc, p_agent_esc=p_esc, p_agent_reopen=p_reo,
             reopen_share_top=dict(reopen_share.head(1).round(3)), bounce_share=bounce_share.round(3).to_dict())

    # ------------------------------------------------------------------ 8. customer experience
    h("8. Customer experience")
    p(f"CSAT response rate {pct(hk['csat_response_rate'])}, mean {hk['csat_mean']:.2f}, share 1-2: {pct((df['csat_score'] <= 2).sum() / df['csat_score'].notna().sum())}")
    for col in ["is_escalated", "eng_handoffs", "reopen_count", "eng_outcome"]:
        table(df.groupby(col)["csat_score"].agg(["mean", "count"]))
    buckets = pd.cut(df["ttr_h"], [0, 8, 24, 72, 168, 1e4], labels=["<8h", "8-24h", "1-3d", "3-7d", ">7d"])
    table(df.groupby(buckets, observed=True)["csat_score"].agg(["mean", "count"]))
    p(f"Reopen rate {pct(hk['reopen_rate'])}; escalated {pct(esc['is_reopened'].mean())} vs L1-only {pct(df.loc[~df['is_escalated'], 'is_reopened'].mean())}")
    cl = df.groupby("client_name").size()
    p(f"{cl.size} clients; tickets per client median {cl.median():.0f}, max {cl.max()} ({cl.idxmax()}); top 20 clients = {pct(cl.nlargest(20).sum() / len(df))} of tickets (no single client dominates).")
    rep = df.sort_values("created_at")
    prev = rep.groupby(["client_id", "category"])["created_at"].shift()
    p(f"Same client + same category again within 7 days: {int(((rep['created_at'] - prev).dt.days < 7).sum())} tickets ({pct(((rep['created_at'] - prev).dt.days < 7).mean())}).")
    J["csat_by_handoffs"] = df.groupby("eng_handoffs")["csat_score"].mean().round(2).to_dict()
    J["csat_by_reopen"] = df.groupby("reopen_count")["csat_score"].mean().round(2).to_dict()

    # ------------------------------------------------------------------ 9. can we predict bounces? (decides ML vs rules)
    h("9. Is a bounce predictable from ticket fields? (decides whether v0 uses ML)")
    e = esc.sort_values("created_at").copy()
    cut = e["created_at"].quantile(0.7)
    tr, te = e[e["created_at"] <= cut], e[e["created_at"] > cut]
    aucs = {}
    for name, cols in {"ticket fields": ["category", "priority", "channel", "client_tier"],
                       "ticket fields + agent": ["category", "priority", "channel", "client_tier", "assigned_agent"]}.items():
        ct = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), cols),
                                ("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "description")])
        model = make_pipeline(ct, LogisticRegression(max_iter=2000, C=0.5)).fit(tr, tr["is_bounce"])
        aucs[name] = roc_auc_score(te["is_bounce"], model.predict_proba(te)[:, 1])
    p(f"Time-split logistic regression (train first 70% of escalations, test last 30%, n_test={len(te)}): "
      + ", ".join(f"{k} AUC={v:.2f}" for k, v in aucs.items())
      + ". AUC ~0.5 = no better than chance. Bounces depend on what is *in* the escalation, which the data does not record. "
        "So v0 uses transparent rules + checklists, not a predictive model.")
    J["bounce_model_auc"] = aucs

    # ------------------------------------------------------------------ 10. known-issue lookup backtest
    h("10. Backtest: would 'related engineering tickets' have surfaced known issues?")
    hits = []
    for _, r in e.iterrows():
        rel = find_related_eng_tickets(df, r["category"], r["description"], as_of=r["created_at"])
        hits.append(not rel.empty)
    e["flag"] = hits
    dup = e["eng_outcome"] == "Duplicate of known issue"
    recall = e.loc[dup, "flag"].mean()
    flag_rate = e["flag"].mean()
    prec = dup[e["flag"]].mean()
    p(f"For each historical escalation, using only tickets created before it (no look-ahead): related ticket shown for {pct(flag_rate)} of escalations; "
      f"{pct(recall)} of eventual 'Duplicate of known issue' escalations would have had a related ticket shown; only {pct(prec)} of shown cases were duplicates "
      f"(base rate {pct(dup.mean())}). Precision equals the base rate (also tested: 7/14/30-day windows, similarity 0.35/0.6/0.9, open-only), "
      "because the same templated descriptions recur constantly. Conclusion: text similarity cannot identify known issues in this data. "
      "v0 shows recent similar escalations as context and never blocks; a real known-issue registry needs engineering to tag incidents (roadmap week 3).")
    J.update(related_recall_dups=recall, related_flag_rate=flag_rate, related_precision_dups=prec)

    # ------------------------------------------------------------------ figures
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    wk = M.weekly_kpis(df)
    full = wk[(wk["week_start"] >= "2026-07-06") & (wk["week_start"] <= "2026-09-21")]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0))
    ax = axes[0]; style(ax)
    ax.plot(full["week_start"], full["created"], color=BLUE, lw=2, marker="o", ms=4, label="Created")
    ax.plot(full["week_start"], full["closed"], color=ORANGE, lw=2, marker="o", ms=4, label="Closed")
    ax.set_title("Tickets per week (full weeks)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    ax = axes[1]; style(ax)
    bls = M.backlog_series(df)
    ax.plot(bls["period_end"], bls["open_tickets"], color=BLUE, lw=2, marker="o", ms=4)
    ax.set_ylim(0, None)
    ax.set_title("Open tickets at week end", fontsize=10, color=INK, loc="left")
    ax.annotate(f"{bls['open_tickets'].iloc[-1]}", (bls["period_end"].iloc[-1], bls["open_tickets"].iloc[-1]),
                textcoords="offset points", xytext=(4, 4), fontsize=8, color=INK2)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    fig.tight_layout(); fig.savefig(FIG / "fig1_volume_backlog.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 2.6)); style(ax); ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    order = ["Resolved at L1", "Fixed by engineering", "Closed by eng without fix", "Bounced back to L1"]
    s = seg.reindex(order)
    y = np.arange(len(order))
    ax.barh(y + 0.2, s["ticket_share"] * 100, height=0.36, color=GRAY, label="Share of tickets")
    ax.barh(y - 0.2, s["backlog_share"] * 100, height=0.36, color=[BLUE, BLUE, BLUE, ORANGE], label="Share of backlog-hours")
    for i, (ts, bs) in enumerate(zip(s["ticket_share"], s["backlog_share"])):
        ax.text(ts * 100 + 0.8, i + 0.2, f"{ts:.0%} of tickets", va="center", fontsize=8, color=INK2)
        ax.text(bs * 100 + 0.8, i - 0.2, f"{bs:.0%} of backlog-hours", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, order, fontsize=9, color=INK)
    ax.set_xlim(0, 80); ax.invert_yaxis()
    ax.set_xlabel("%", fontsize=8, color=INK2)
    ax.set_title("Bounced escalations: 12% of tickets, 35% of the time tickets spend open", fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "fig2_backlog_hours.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.6))
    hs = df.groupby("eng_handoffs").agg(ttr=("ttr_h", "median"), csat=("csat_score", "mean"))
    labels = ["0 (L1 only)", "1", "2", "3"]
    ax = axes[0]; style(ax)
    ax.bar(labels, hs["ttr"] / 24, color=[GRAY, BLUE, ORANGE, ORANGE], width=0.6)
    for i, v in enumerate(hs["ttr"] / 24):
        ax.text(i, v + 0.1, f"{v:.1f}d", ha="center", fontsize=8, color=INK2)
    ax.set_title("Median days to final close, by eng handoffs", fontsize=10, color=INK, loc="left")
    ax = axes[1]; style(ax)
    ax.bar(labels, hs["csat"], color=[GRAY, BLUE, ORANGE, ORANGE], width=0.6)
    for i, v in enumerate(hs["csat"]):
        ax.text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=8, color=INK2)
    ax.set_ylim(0, 5)
    ax.set_title("Mean CSAT (1-5), by eng handoffs", fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "fig3_handoffs_ttr_csat.png", dpi=200); plt.close(fig)

    (DOCS / "analysis_results.md").write_text("# RideOps diagnosis: computed results\n\nGenerated by `analysis/run_analysis.py`.\n" + "\n".join(out), encoding="utf-8")
    (DOCS / "metrics.json").write_text(json.dumps(J, indent=2, default=str), encoding="utf-8")
    wk.to_csv(DOCS / "weekly_kpis.csv", index=False)
    print("\n".join(out))


if __name__ == "__main__":
    main()
