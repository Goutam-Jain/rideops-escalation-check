"""Escalation readiness check: related-ticket lookup, readiness verdict, handoff note, audit log."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .checklists import checklist_for

PLACEHOLDERS = {"na", "n/a", "-", "--", "tbd", "?", "none", "nil", "unknown", "x", "."}
MIN_ANSWER_CHARS = 3


# --------------------------------------------------------------------------- related tickets
def find_related_eng_tickets(df: pd.DataFrame, category: str, description: str,
                             as_of: pd.Timestamp, window_days: int = 14,
                             min_similarity: float = 0.6, top_k: int = 5,
                             exclude_ticket: str | None = None) -> pd.DataFrame:
    """Escalated tickets in the same category from the last `window_days` whose
    description is textually similar (TF-IDF cosine on word uni/bi-grams).

    This is retrieval, not prediction: it shows the agent what engineering has
    recently seen and what happened to it, so they can reference or link an
    existing engineering ticket. The agent decides; it never blocks.
    Backtest (analysis/run_analysis.py s.10): a match does NOT reliably mean
    "known issue" - precision equals the base rate - so it is context only.
    """
    if not description or not description.strip():
        return pd.DataFrame()
    pool = df[
        df["is_escalated"]
        & (df["category"] == category)
        & (df["created_at"] < as_of)
        & (df["created_at"] >= as_of - pd.Timedelta(days=window_days))
    ]
    if exclude_ticket:
        pool = pool[pool["ticket_id"] != exclude_ticket]
    if pool.empty:
        return pd.DataFrame()

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        matrix = vec.fit_transform(pool["description"].tolist() + [description])
    except ValueError:  # query and pool contain only stop words
        return pd.DataFrame()
    sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    # Open *as of* the check time (not the snapshot status) so backtests do not peek ahead.
    open_then = pool["final_closed_at"].isna() | (pool["final_closed_at"] > as_of)
    out = pool.assign(similarity=sims, open_then=open_then.values)
    out = out[out["similarity"] >= min_similarity]
    cols = ["ticket_id", "eng_ticket_id", "created_at", "description", "eng_outcome", "status", "open_then", "similarity"]
    return out.sort_values(["similarity", "created_at"], ascending=[False, False])[cols].head(top_k)


# --------------------------------------------------------------------------- readiness
@dataclass
class Readiness:
    verdict: str                     # READY | NOT_READY | REVIEW_ROUTE
    missing_info: list[str] = field(default_factory=list)
    missing_checks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    completeness: float = 0.0        # share of checklist items satisfied, 0..1

    @property
    def ready(self) -> bool:
        return self.verdict == "READY"

    def set_completeness(self, total: int) -> None:
        missing = len(self.missing_info) + len(self.missing_checks)
        self.completeness = 1.0 if total == 0 else (total - missing) / total


def _is_answered(value: str | None) -> bool:
    if value is None:
        return False
    v = value.strip()
    return len(v) >= MIN_ANSWER_CHARS and v.lower() not in PLACEHOLDERS


def evaluate_readiness(category: str, answers: dict[str, str], checks: dict[str, bool],
                       related: pd.DataFrame | None = None) -> Readiness:
    """Rule-based, fully explainable verdict.

    NOT_READY    - any required info blank/placeholder, or any check not ticked
    REVIEW_ROUTE - complete, but the category is usually not an engineering issue
    READY        - complete; an open related engineering ticket adds a note to link it
    """
    spec = checklist_for(category)
    r = Readiness(verdict="READY")
    r.missing_info = [label for key, label in spec["info"].items() if not _is_answered(answers.get(key))]
    r.missing_checks = [label for key, label in spec["checks"].items() if not checks.get(key)]
    r.set_completeness(len(spec["info"]) + len(spec["checks"]))

    if related is not None and not related.empty:
        open_related = related[related["open_then"]] if "open_then" in related else related[related["status"] != "Closed"]
        if not open_related.empty:
            ids = ", ".join(open_related["eng_ticket_id"].astype(str).head(3))
            r.notes.append(f"Similar escalation(s) still open with engineering ({ids}). If it is the same problem, reference them in the note so engineering doesn't start from zero.")
        if related["eng_outcome"].eq("Duplicate of known issue").any():
            r.notes.append("A similar recent escalation was closed as 'Duplicate of known issue'. Worth checking whether this is the same issue.")

    if r.missing_info or r.missing_checks:
        r.verdict = "NOT_READY"
    elif spec.get("route"):
        r.verdict = "REVIEW_ROUTE"
        r.notes.insert(0, spec["route"])
    return r


# --------------------------------------------------------------------------- handoff note
def build_escalation_note(ticket: dict, category: str, answers: dict[str, str],
                          checks: dict[str, bool], readiness: Readiness,
                          related: pd.DataFrame | None = None, agent: str = "") -> str:
    """Markdown note the agent pastes into the engineering ticket."""
    spec = checklist_for(category)
    lines = [
        f"### Escalation: {ticket.get('ticket_id', 'NEW')} | {category} | {ticket.get('priority', '')}",
        f"**Client:** {ticket.get('client_name', '')} ({ticket.get('client_tier', '')})  ",
        f"**Summary:** {ticket.get('description', '')}  ",
        f"**Raised by L1:** {agent}  ",
        f"**Checklist completeness:** {readiness.completeness:.0%}",
        "",
        "#### Information collected",
    ]
    for key, label in spec["info"].items():
        val = (answers.get(key) or "").strip() or "_not provided_"
        lines.append(f"- **{label}:** {val}")
    lines += ["", "#### L1 checks performed"]
    for key, label in spec["checks"].items():
        lines.append(f"- [{'x' if checks.get(key) else ' '}] {label}")
    if related is not None and not related.empty:
        lines += ["", "#### Possibly related engineering tickets"]
        for _, row in related.iterrows():
            outcome = row["eng_outcome"] if isinstance(row["eng_outcome"], str) else "in progress"
            lines.append(f"- {row['eng_ticket_id']} ({row['status']}, {outcome}): {row['description']}")
    if readiness.notes:
        lines += ["", "#### Notes"] + [f"- {n}" for n in readiness.notes]
    return "\n".join(lines)


# --------------------------------------------------------------------------- audit log
LOG_FIELDS = ["logged_at", "ticket_id", "agent", "category", "priority", "verdict",
              "completeness", "missing_info", "missing_checks", "related_eng_tickets", "action"]


def log_check(path: Path, ticket_id: str, agent: str, category: str, priority: str,
              readiness: Readiness, related: pd.DataFrame | None, action: str) -> None:
    """Append one row per readiness check. This log is what makes the leading
    KPI ('% of escalations sent with a complete note') measurable in a pilot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    row = {
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "ticket_id": ticket_id,
        "agent": agent,
        "category": category,
        "priority": priority,
        "verdict": readiness.verdict,
        "completeness": round(readiness.completeness, 3),
        "missing_info": len(readiness.missing_info),
        "missing_checks": len(readiness.missing_checks),
        "related_eng_tickets": ";".join(related["eng_ticket_id"].astype(str)) if related is not None and not related.empty else "",
        "action": action,
    }
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def read_log(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LOG_FIELDS)
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=LOG_FIELDS)
