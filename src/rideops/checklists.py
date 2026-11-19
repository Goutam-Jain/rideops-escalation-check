"""Pre-escalation checklists, one per category.

IMPORTANT, read before changing anything: these lists are NOT derived from the
dataset. The dataset only tells us *that* 36% of escalations bounce back for
"Insufficient info" or "Config issue"; it does not say which information was
missing. These are a v0 starting draft based on what an engineer would need to
reproduce each issue type. Week 1 of the plan is to review them with
Engineering (Sandeep) and adjust. They live in one file on purpose so that
review is a one-file change.

Each category has:
  info    - facts the agent must capture (free text, required, end up in the note)
  checks  - things the agent must confirm they did/verified (checkboxes)
  route   - optional note when the category usually belongs somewhere other
            than engineering (shown as guidance, not a block)
"""
from __future__ import annotations

UNIVERSAL_INFO = {
    "steps": "What the user did / steps to reproduce",
    "expected_actual": "Expected vs actual result",
    "first_seen": "When it started (date/time, IST)",
    "impact": "Who is affected (how many employees / sites / trips)",
}
UNIVERSAL_CHECKS = {
    "known_issue_checked": "Searched recent engineering tickets for the same issue (see 'Recent similar escalations')",
    "evidence_attached": "Screenshot / recording / export attached to the ticket",
}

CHECKLISTS: dict[str, dict] = {
    "Roster / booking not syncing": {
        "info": {
            "booking_ids": "Booking / cancellation IDs and the time each change was made",
            "shift": "Shift date and time affected",
            "employee_ids": "Affected employee IDs (or count + sample IDs)",
        },
        "checks": {
            "cutoff_config": "Checked the client's booking-cutoff and shift settings in admin config",
            "roster_time": "Confirmed the change was made before the roster-generation cutoff",
            "admin_vs_app": "Compared the admin view with the employee-app view",
        },
    },
    "Live tracking / GPS incorrect": {
        "info": {
            "trip_id": "Trip ID(s)",
            "driver_vehicle": "Driver ID and vehicle number",
            "app_version": "Driver app version and phone OS",
            "time_window": "Exact time window of the wrong location",
        },
        "checks": {
            "gps_permission": "Confirmed driver app has location permission set to 'Always' and data is on",
            "scope": "Checked whether one cab or many cabs are affected",
            "link_access": "If a tracking link: confirmed link validity and viewer access settings",
        },
    },
    "Report export failing": {
        "info": {
            "report_name": "Report name, format and filters used",
            "date_range": "Date range requested",
            "error": "Error message or behaviour (blank file, stuck loading, missing rows)",
        },
        "checks": {
            "retry_smaller": "Retried with a shorter date range / different format",
            "user_role": "Confirmed the user's role has access to this report",
        },
    },
    "OTP / boarding issue": {
        "info": {
            "trip_id": "Trip ID and employee ID",
            "otp_detail": "OTP shown to employee vs OTP entered by driver",
            "app_version": "Driver app version and phone OS",
        },
        "checks": {
            "network": "Checked driver/employee network status at the time",
            "sms_delivery": "Checked OTP SMS/notification delivery status",
        },
    },
    "Billing / invoice discrepancy": {
        "info": {
            "invoice": "Invoice number and billing period",
            "line_items": "Disputed line items with client's expected value vs invoiced value",
        },
        "checks": {
            "rate_card": "Checked which rate card / vendor contract version was applied",
            "trip_data": "Cross-checked disputed trips against trip logs",
        },
    },
    "Employee address / geocode change": {
        "info": {
            "employee_ids": "Employee IDs, old and new address",
            "update_time": "When the address was updated and when routing last ran",
        },
        "checks": {
            "pin_verified": "Verified the saved map pin (lat/long) matches the new address",
            "routing_rerun": "Confirmed routing has run since the address was updated",
        },
    },
    "Cab delay / no-show complaint": {
        "info": {
            "trip_id": "Trip ID, scheduled vs actual pickup time",
            "driver_vendor": "Driver and vendor",
        },
        "checks": {
            "ops_checked": "Checked with transport ops / vendor whether this was an operational delay",
            "app_mismatch": "Confirmed the app data itself is wrong (not just a late cab)",
        },
        "route": "Most cab-delay complaints are operational (vendor/driver). Escalate to engineering only if app data is wrong.",
    },
    "Login / access issue": {
        "info": {
            "user": "User ID / email and role",
            "error": "Exact error message",
        },
        "checks": {
            "provisioned": "Confirmed the user is provisioned and has the right role assigned",
            "reset_tried": "Tried password reset / re-invite",
        },
    },
    "How-to / feature request": {
        "info": {"request": "What the client is trying to do"},
        "checks": {
            "docs_checked": "Checked help docs / FAQ for an existing way to do this",
        },
        "route": "How-to questions are answered by L1; feature requests go to the product backlog, not an engineering escalation.",
    },
    "Driver / vehicle compliance docs": {
        "info": {
            "driver": "Driver / vehicle ID and document type",
            "error": "What fails (upload, expiry date, approval)",
        },
        "checks": {
            "file_valid": "Checked file format/size and the document expiry date",
        },
    },
}

DEFAULT_CHECKLIST = {"info": {}, "checks": {}}


def checklist_for(category: str) -> dict:
    """Merged checklist (universal + category-specific) for a category."""
    spec = CHECKLISTS.get(category, DEFAULT_CHECKLIST)
    return {
        "info": {**UNIVERSAL_INFO, **spec.get("info", {})},
        "checks": {**UNIVERSAL_CHECKS, **spec.get("checks", {})},
        "route": spec.get("route"),
    }
