"""Loading, validating and enriching the RideOps support-ticket export.

Everything downstream (analysis script, Streamlit app, tests) goes through
`load_tickets()`, so the cleaning rules live in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Union

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "ticket_id", "created_at", "channel", "client_id", "client_name", "client_tier",
    "category", "priority", "description", "assigned_agent", "first_response_at",
    "escalated_to_eng", "eng_ticket_id", "eng_handoffs", "eng_outcome", "status",
    "first_closed_at", "final_closed_at", "reopen_count", "csat_score",
]
TIMESTAMP_COLUMNS = ["created_at", "first_response_at", "first_closed_at", "final_closed_at"]

# Engineering outcomes, grouped by what they mean for the escalation process.
OUTCOME_FIXED = "Fixed"
BOUNCE_OUTCOMES = {  # engineering sent it back to L1 to do more work
    "Config issue - returned to L1",
    "Insufficient info - returned to L1",
}
NON_FIX_OUTCOMES = {  # engineering closed it without a code fix
    "Duplicate of known issue",
    "Not a bug - how-to",
    "Not reproducible",
}
KNOWN_OUTCOMES = {OUTCOME_FIXED} | BOUNCE_OUTCOMES | NON_FIX_OUTCOMES

# The data was pulled on 30 Sep 2026 (per the brief). Open tickets are aged to
# the end of that day. Overridable for a newer export.
DEFAULT_SNAPSHOT = pd.Timestamp("2026-10-01 00:00")

DataSource = Union[str, Path, IO]


class DataValidationError(ValueError):
    """Raised when the CSV cannot be used safely (missing columns, bad values)."""


@dataclass
class ValidationReport:
    rows: int
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


def _hours(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.total_seconds() / 3600.0


def validate_raw(df: pd.DataFrame) -> ValidationReport:
    """Hard-fail on structural problems; collect soft data-quality warnings."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"CSV is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise DataValidationError("CSV has no rows.")
    if df["ticket_id"].duplicated().any():
        dups = df.loc[df["ticket_id"].duplicated(), "ticket_id"].head(5).tolist()
        raise DataValidationError(f"Duplicate ticket_id values, e.g. {dups}")

    report = ValidationReport(rows=len(df))
    bad_esc = ~df["escalated_to_eng"].isin(["Yes", "No"])
    if bad_esc.any():
        report.warnings.append(f"{bad_esc.sum()} rows have escalated_to_eng not in Yes/No.")
    unknown = set(df["eng_outcome"].dropna()) - KNOWN_OUTCOMES
    if unknown:
        report.warnings.append(f"Unknown eng_outcome values: {sorted(unknown)}")
    return report


def _check_consistency(df: pd.DataFrame, report: ValidationReport) -> None:
    """Soft checks on the parsed frame. Warnings only, never silent fixes."""
    checks = {
        "first_response_at before created_at": df["first_response_at"] < df["created_at"],
        "first_closed_at before created_at": df["first_closed_at"] < df["created_at"],
        "final_closed_at before first_closed_at": df["final_closed_at"] < df["first_closed_at"],
        "status Closed but final_closed_at blank": (df["status"] == "Closed") & df["final_closed_at"].isna(),
        "status Open but final_closed_at set": (df["status"] != "Closed") & df["final_closed_at"].notna(),
        "escalated=Yes with no eng_ticket_id": df["is_escalated"] & df["eng_ticket_id"].isna(),
        "escalated=No with eng_handoffs > 0": ~df["is_escalated"] & (df["eng_handoffs"] > 0),
        "reopen_count > 0 but first_closed_at blank": (df["reopen_count"] > 0) & df["first_closed_at"].isna(),
    }
    for label, mask in checks.items():
        n = int(mask.sum())
        if n:
            report.warnings.append(f"{n} ticket(s): {label}")
    unparsed = df.attrs.get("unparsed_timestamps", 0)
    if unparsed:
        report.warnings.append(f"{unparsed} non-blank timestamp value(s) could not be parsed and were set blank")


def load_tickets(source: DataSource, snapshot: pd.Timestamp | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    """Read the ticket CSV, validate it and add derived columns.

    Blanks are meaningful, not errors:
      * eng_ticket_id / eng_outcome blank  -> ticket was never escalated
      * first_response_at blank            -> no L1 reply yet (all are recent, open tickets)
      * first/final_closed_at blank        -> ticket still open (final blank = open now)
      * csat_score blank                   -> client did not rate (not a zero)
    """
    try:
        raw = pd.read_csv(source)
    except Exception as exc:  # pandas raises several error types for bad files
        raise DataValidationError(f"Could not read CSV: {exc}") from exc

    report = validate_raw(raw)
    df = raw.copy()

    unparsed = 0
    for col in TIMESTAMP_COLUMNS:
        parsed = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M", errors="coerce")
        retry = parsed.isna() & df[col].notna()
        if retry.any():  # e.g. seconds included in a newer export
            parsed[retry] = pd.to_datetime(df.loc[retry, col], format="ISO8601", errors="coerce")
        unparsed += int((parsed.isna() & df[col].notna()).sum())
        df[col] = parsed
    df.attrs["unparsed_timestamps"] = unparsed

    df["eng_handoffs"] = pd.to_numeric(df["eng_handoffs"], errors="coerce").fillna(0).astype(int)
    df["reopen_count"] = pd.to_numeric(df["reopen_count"], errors="coerce").fillna(0).astype(int)
    df["csat_score"] = pd.to_numeric(df["csat_score"], errors="coerce")

    snap = snapshot if snapshot is not None else DEFAULT_SNAPSHOT
    df.attrs["snapshot"] = snap

    df["is_escalated"] = df["escalated_to_eng"].eq("Yes")
    df["is_open"] = df["status"].ne("Closed")
    df["is_bounce"] = df["eng_outcome"].isin(BOUNCE_OUTCOMES)
    df["is_fixed"] = df["eng_outcome"].eq(OUTCOME_FIXED)
    df["is_reopened"] = df["reopen_count"] > 0
    df["extra_handoffs"] = (df["eng_handoffs"] - 1).clip(lower=0)

    df["frt_h"] = _hours(df["first_response_at"], df["created_at"])
    df["ttfc_h"] = _hours(df["first_closed_at"], df["created_at"])
    df["ttr_h"] = _hours(df["final_closed_at"], df["created_at"])
    # Time in system: closed -> resolution time; open -> age at snapshot.
    end = df["final_closed_at"].fillna(snap)
    df["time_in_system_h"] = _hours(end, df["created_at"])
    df["age_h"] = np.where(df["is_open"], _hours(pd.Series(snap, index=df.index), df["created_at"]), np.nan)

    df["created_week"] = df["created_at"].dt.to_period("W-SUN").dt.start_time
    df["created_month"] = df["created_at"].dt.to_period("M").astype(str)
    df["escalation_segment"] = np.select(
        [~df["is_escalated"], df["is_bounce"], df["is_fixed"]],
        ["Resolved at L1", "Bounced back to L1", "Fixed by engineering"],
        default="Closed by eng without fix",
    )

    _check_consistency(df, report)
    return df, report
