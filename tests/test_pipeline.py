"""
tests/test_pipeline.py -- run with `pytest tests/ -v`

Small, focused tests -- one idea per test.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.agent import build_context, check_account_history, generate_alert, tag_patterns
from src.alerter import format_alert, select_alert_candidates
from src.detector import classify_severity


# --------------------------------------------------------------------------
# Rule-tagging (the "fraud code lookup")
# --------------------------------------------------------------------------
def test_tag_patterns_flags_large_amount_deviation():
    row = pd.Series({"spend_ratio": 5.0, "txn_count_1h": 1, "is_night": False, "merchant_mismatch": 0})
    assert "Large Amount Deviation" in tag_patterns(row)


def test_tag_patterns_flags_burst():
    row = pd.Series({"spend_ratio": 1.0, "txn_count_1h": 4, "is_night": False, "merchant_mismatch": 0})
    assert "Rapid Transaction Burst" in tag_patterns(row)


def test_tag_patterns_flags_night_and_unfamiliar_merchant_together():
    row = pd.Series({"spend_ratio": 1.0, "txn_count_1h": 1, "is_night": True, "merchant_mismatch": 1})
    tags = tag_patterns(row)
    assert "Off-Hours Activity" in tags
    assert "Unfamiliar Merchant Category" in tags


def test_tag_patterns_falls_back_when_nothing_fires():
    row = pd.Series({"spend_ratio": 1.0, "txn_count_1h": 1, "is_night": False, "merchant_mismatch": 0})
    assert tag_patterns(row) == ["General Statistical Anomaly"]


# --------------------------------------------------------------------------
# Severity classification
# --------------------------------------------------------------------------
def test_classify_severity_thresholds():
    assert classify_severity(0.9) == "Red"
    assert classify_severity(0.5) == "Yellow"
    assert classify_severity(0.1) == "Green"


# --------------------------------------------------------------------------
# Account history / cooldown
# --------------------------------------------------------------------------
def test_no_history_means_safe_to_dispatch():
    safe, reason, count = check_account_history("ACC001", pd.Timestamp("2025-03-01"), {})
    assert safe is True
    assert count == 0


def test_cooldown_blocks_a_second_alert_within_an_hour():
    last = pd.Timestamp("2025-03-01 10:00:00")
    history = {"ACC001": [last]}
    now = last + pd.Timedelta(minutes=20)
    safe, reason, _ = check_account_history("ACC001", now, history)
    assert safe is False
    assert "hour" in reason.lower()


def test_cooldown_clears_after_an_hour():
    last = pd.Timestamp("2025-03-01 10:00:00")
    history = {"ACC001": [last]}
    now = last + pd.Timedelta(hours=2)
    safe, _, _ = check_account_history("ACC001", now, history)
    assert safe is True


# --------------------------------------------------------------------------
# The safety boundary: dispatch_safe is always a Python decision
# --------------------------------------------------------------------------
def test_green_severity_is_never_dispatch_safe():
    row = pd.Series({
        "txn_id": "TXN1", "account_id": "ACC001", "account_name": "Test", "timestamp": pd.Timestamp("2025-03-01"),
        "amount": 50.0, "merchant_category": "groceries", "severity": "Green", "anomaly_score": 0.1,
        "spend_ratio": 1.0, "txn_count_1h": 1, "is_night": False, "merchant_mismatch": 0,
    })
    ctx = build_context(row, {})
    assert ctx.dispatch_safe is False


def test_alert_dispatch_safe_matches_layer_a_even_without_llm():
    """No Ollama server running in the test environment, so this exercises the template fallback."""
    row = pd.Series({
        "txn_id": "TXN2", "account_id": "ACC001", "account_name": "Test", "timestamp": pd.Timestamp("2025-03-01"),
        "amount": 500.0, "merchant_category": "crypto_exchange", "severity": "Red", "anomaly_score": 0.95,
        "spend_ratio": 8.0, "txn_count_1h": 1, "is_night": True, "merchant_mismatch": 1,
    })
    ctx = build_context(row, {})
    assert ctx.dispatch_safe is True
    alert = generate_alert(ctx)
    assert alert["dispatch_safe"] == ctx.dispatch_safe
    assert alert["generated_by"] == "template"
    for key in ["headline", "explanation", "recommended_action", "urgency", "escalation_required"]:
        assert key in alert


# --------------------------------------------------------------------------
# Burst de-duplication
# --------------------------------------------------------------------------
def test_select_alert_candidates_keeps_only_highest_severity_per_hour():
    base = pd.Timestamp("2025-03-01 10:00:00")
    df = pd.DataFrame([
        {"account_id": "ACC001", "timestamp": base, "severity": "Yellow", "anomaly_score": 0.5, "txn_id": "A"},
        {"account_id": "ACC001", "timestamp": base + pd.Timedelta(minutes=5), "severity": "Red", "anomaly_score": 0.9, "txn_id": "B"},
        {"account_id": "ACC001", "timestamp": base + pd.Timedelta(minutes=10), "severity": "Yellow", "anomaly_score": 0.6, "txn_id": "C"},
    ])
    out = select_alert_candidates(df)
    assert len(out) == 1
    assert out.iloc[0]["txn_id"] == "B"


# --------------------------------------------------------------------------
# Console formatting
# --------------------------------------------------------------------------
def test_format_alert_includes_key_fields():
    alert = {
        "txn_id": "TXN1", "account_name": "Test Account", "amount": 100.0, "severity": "Red",
        "anomaly_score": 0.9, "urgency": "high", "tags": ["Large Amount Deviation"],
        "headline": "Big spike alert", "explanation": "It was big.", "recommended_action": "Check it.",
        "dispatch_safe": True, "dispatch_reason": "ok", "generated_by": "template",
    }
    text = format_alert(alert)
    assert "Test Account" in text
    assert "[RED]" in text
    assert "Large Amount Deviation" in text


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
