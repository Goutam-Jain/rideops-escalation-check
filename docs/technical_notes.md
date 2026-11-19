# Technical notes: Escalation Readiness Check

## What the app does

| Tab | For | What happens |
|---|---|---|
| **Escalate a ticket** | L1 agent | Your open L1 queue (tickets sent back by engineering listed first), or a new ticket. It shows what happened to past escalations of this kind and recent similar escalations, then runs a category checklist that returns **Not ready / Ready / Check routing**. You get a structured escalation note to paste into the engineering ticket. Every check is logged. |
| **Escalation quality (weekly)** | Agent / lead | Primary KPI (share of escalations sent back to L1), handoffs per escalation, guardrails (escalation rate, reopen rate), created vs closed, and open backlog. Filter by agent and category. Pilot-log summary. |
| **How this works** | Anyone | The logic, the data rules and the limitations, computed live from the loaded file. |

## Project layout

```
app.py                     Streamlit UI
src/rideops/data.py        load + validate + derive fields (one place for cleaning rules)
src/rideops/metrics.py     KPIs, backlog reconstruction, Little's-law decomposition, agent mix-adjustment
src/rideops/checklists.py  DRAFT per-category checklists (to be reviewed with engineering)
src/rideops/escalation.py  similar-ticket retrieval, readiness verdict, escalation note, audit log
analysis/run_analysis.py   reproducible diagnosis → docs/analysis_results.md, docs/metrics.json, figures
tests/test_core.py         validation, metric consistency, verdict rules, no look-ahead, log round-trip
docs/                      write-up (PDF/MD), slide outline, interview prep, computed results
data/                      ticket CSV + data dictionary; escalation_checks.csv is created at runtime
```

## Logic and limitations (read before demoing)

- **Verdict = transparent rules.** Any required field that is blank or a placeholder ("n/a", "-", "tbd"…), or any check left unticked, gives *Not ready*. Complete gives *Ready*. Complete in a category that is usually not an engineering issue (how-to, cab delay) gives *Check routing*.
- **No prediction, on purpose.** A logistic-regression model on category, priority, channel, tier, agent and summary text predicted bounces no better than chance (time-split AUC 0.48–0.52). See `analysis/run_analysis.py` §9.
- **"Recent similar escalations" is context only.** It shows escalations in the same category from the last 14 days with TF-IDF cosine ≥ 0.6. In a no-look-ahead backtest, a match did not identify true "Duplicate of known issue" cases better than the base rate (§10), so it never blocks.
- **Checklists are a draft.** The data shows *that* escalations come back for missing info or config, not *which* info. Week 1 of the plan is reviewing them with engineering.
- **Data rules.** Blank `eng_outcome` = never escalated. Blank `final_closed_at` = still open. Blank `csat_score` = not rated. "Open at time t" ignores reopen gaps (reopen times aren't recorded), which makes it a slight upper bound.
- The log is a local CSV (`data/escalation_checks.csv`). On a hosted demo with ephemeral storage it resets. A real pilot would write to the helpdesk or a database.
- **This is historical data.** The tool has not been shown to reduce the backlog. That needs the pilot described in the write-up.
