import io
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rideops import metrics as M  # noqa: E402
from rideops.checklists import CHECKLISTS, checklist_for  # noqa: E402
from rideops.data import DataValidationError, load_tickets  # noqa: E402
from rideops.escalation import (  # noqa: E402
    build_escalation_note, evaluate_readiness, find_related_eng_tickets, log_check, read_log,
)

DATA = ROOT / "data" / "rideops_support_tickets.csv"


@pytest.fixture(scope="module")
def df():
    frame, _ = load_tickets(DATA)
    return frame


# ---------------------------------------------------------------- loading / validation
def test_loads_full_dataset(df):
    assert len(df) == 1444
    assert df["ticket_id"].is_unique


def test_missing_column_is_rejected():
    raw = pd.read_csv(DATA).drop(columns=["eng_outcome"])
    with pytest.raises(DataValidationError, match="eng_outcome"):
        load_tickets(io.StringIO(raw.to_csv(index=False)))


def test_duplicate_ids_are_rejected():
    raw = pd.read_csv(DATA)
    raw = pd.concat([raw, raw.head(1)])
    with pytest.raises(DataValidationError, match="Duplicate"):
        load_tickets(io.StringIO(raw.to_csv(index=False)))


def test_garbage_file_is_rejected():
    with pytest.raises(DataValidationError):
        load_tickets(io.StringIO("not,a,ticket,file\n1,2,3,4\n"))


def test_unparseable_timestamp_becomes_warning_not_crash():
    raw = pd.read_csv(DATA)
    raw.loc[0, "created_at"] = "not a date"
    frame, report = load_tickets(io.StringIO(raw.to_csv(index=False)))
    assert pd.isna(frame.loc[0, "created_at"])
    assert any("could not be parsed" in w for w in report.warnings)


def test_blank_semantics(df):
    # never-escalated tickets have no outcome; open tickets have no final close
    assert df.loc[~df["is_escalated"], "eng_outcome"].isna().all()
    assert df.loc[df["is_open"], "final_closed_at"].isna().all()
    # every bounce needed at least two handoffs
    assert (df.loc[df["is_bounce"], "eng_handoffs"] >= 2).all()


# ---------------------------------------------------------------- metrics
def test_backlog_matches_brief(df):
    """Brief: 'open tickets are up about 60% since July'."""
    jul = M.open_count_at(df, pd.Timestamp("2026-07-31 23:59"))
    sep = M.open_count_at(df, pd.Timestamp("2026-09-30 23:59"))
    assert sep == int(df["is_open"].sum())
    assert 0.5 <= sep / jul - 1 <= 0.7


def test_headline_kpis_consistent(df):
    k = M.headline_kpis(df)
    assert k["open_now"] == k["open_with_l1"] + k["open_with_eng"]
    assert 0 < k["bounce_rate"] < 1
    assert k["median_ttr_escalated_h"] > k["median_ttr_l1_h"]


def test_backlog_shares_sum_to_one(df):
    seg = M.backlog_hours_by(df, "escalation_segment")
    assert seg["backlog_share"].sum() == pytest.approx(1.0)
    assert seg["ticket_share"].sum() == pytest.approx(1.0)


def test_weekly_kpis_cover_all_tickets(df):
    wk = M.weekly_kpis(df)
    assert wk["created"].sum() == len(df)


def test_profile_falls_back_to_category_for_rare_description(df):
    prof = M.category_escalation_profile(df, "Login / access issue", "something nobody wrote")
    assert prof["scope"] == "category"


# ---------------------------------------------------------------- escalation logic
def test_every_category_has_a_checklist(df):
    assert set(df["category"].unique()) <= set(CHECKLISTS)


def test_blank_form_is_not_ready():
    r = evaluate_readiness("Roster / booking not syncing", {}, {})
    assert r.verdict == "NOT_READY"
    assert r.completeness == 0


def test_placeholder_answers_do_not_count():
    spec = checklist_for("Login / access issue")
    answers = {k: "n/a" for k in spec["info"]}
    checks = {k: True for k in spec["checks"]}
    r = evaluate_readiness("Login / access issue", answers, checks)
    assert r.verdict == "NOT_READY"
    assert len(r.missing_info) == len(spec["info"])


def test_complete_form_is_ready_and_note_contains_answers():
    cat = "Login / access issue"
    spec = checklist_for(cat)
    answers = {k: f"value for {k}" for k in spec["info"]}
    checks = {k: True for k in spec["checks"]}
    r = evaluate_readiness(cat, answers, checks)
    assert r.verdict == "READY" and r.completeness == 1
    note = build_escalation_note({"ticket_id": "SUP-1", "priority": "P2"}, cat, answers, checks, r)
    assert "value for user" in note and "[x]" in note


def test_route_categories_need_review():
    cat = "How-to / feature request"
    spec = checklist_for(cat)
    r = evaluate_readiness(cat, {k: "filled in" for k in spec["info"]}, {k: True for k in spec["checks"]})
    assert r.verdict == "REVIEW_ROUTE"


def test_related_tickets_never_look_ahead(df):
    as_of = pd.Timestamp("2026-08-15")
    rel = find_related_eng_tickets(df, "Roster / booking not syncing",
                                   "Cancelled bookings still appearing in tomorrow's roster", as_of=as_of)
    assert not rel.empty
    assert (rel["created_at"] < as_of).all()
    assert (rel["created_at"] >= as_of - pd.Timedelta(days=14)).all()


def test_related_tickets_empty_query(df):
    assert find_related_eng_tickets(df, "Report export failing", "   ", as_of=pd.Timestamp("2026-09-01")).empty


def test_log_roundtrip(tmp_path):
    r = evaluate_readiness("Login / access issue", {}, {})
    path = tmp_path / "log.csv"
    log_check(path, "SUP-1", "Agent", "Login / access issue", "P3", r, None, "checked")
    log_check(path, "SUP-1", "Agent", "Login / access issue", "P3", r, None, "escalated")
    log = read_log(path)
    assert list(log["action"]) == ["checked", "escalated"]
