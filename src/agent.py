"""
The two-layer agent.

Layer A (Python rules engine -- deterministic, no LLM):
  - tag_patterns(): looks at the same 4 features the detector used and
    tags WHY a transaction looks suspicious, in plain rule form
    (e.g. spend_ratio >= 3 -> "Large Amount Deviation"). This is the
    "fraud code lookup" -- a rules engine, not a black box.
  - check_account_history(): has this account already had an alert
    dispatched recently? Used to avoid spamming the same account.
  - dispatch_safe: a plain Python boolean. The LLM is never asked to
    decide this -- it only ever writes the narrative around a decision
    that's already been made.

Layer B (local Llama call via Ollama, with a template fallback):
  - Takes the Layer A context and writes a short, structured, plain-English
    alert using a local Llama model served by Ollama. If Ollama isn't
    installed, isn't running, or the model hasn't been pulled yet, a
    deterministic template produces the same fields instead -- the
    pipeline never breaks because the LLM isn't available. No API key,
    no network call, no cost -- everything runs on your machine.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import timedelta

ALERT_COOLDOWN = timedelta(hours=1)
REQUIRED_KEYS = ["headline", "explanation", "recommended_action", "urgency", "escalation_required"]
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

SYSTEM_PROMPT = """You are a fraud alert writer for a bank's fraud team.
You will receive a JSON object describing one flagged transaction and the
rule-based tags that triggered it. Reply with ONLY a JSON object (no
markdown, no extra text) with exactly these keys:

  "headline": string, <= 12 words
  "explanation": string, 1-2 plain-English sentences on what happened and why it was flagged
  "recommended_action": string, one concrete next step for the analyst
  "urgency": one of "low", "medium", "high"
  "escalation_required": boolean

Do not include a "dispatch_safe" key -- that decision is made upstream, not by you.
Only use facts present in the input JSON."""


@dataclass
class AlertContext:
    txn_id: str
    account_id: str
    account_name: str
    timestamp: str
    amount: float
    merchant_category: str
    severity: str
    anomaly_score: float
    tags: list[str]
    recent_alerts_for_account: int
    dispatch_safe: bool
    dispatch_reason: str = field(default="")


# --------------------------------------------------------------------------
# Layer A
# --------------------------------------------------------------------------
def tag_patterns(row) -> list[str]:
    """The rules-engine 'fraud code lookup': tags a flagged transaction based on which rule(s) fired."""
    tags = []
    if row["spend_ratio"] >= 3:
        tags.append("Large Amount Deviation")
    if row["txn_count_1h"] >= 3:
        tags.append("Rapid Transaction Burst")
    if row["is_night"]:
        tags.append("Off-Hours Activity")
    if row["merchant_mismatch"]:
        tags.append("Unfamiliar Merchant Category")
    if not tags:
        tags.append("General Statistical Anomaly")
    return tags


def check_account_history(account_id: str, timestamp, dispatched_history: dict) -> tuple[bool, str, int]:
    """Cooldown check: don't dispatch a second alert for the same account within the cooldown window."""
    history = dispatched_history.get(account_id, [])
    recent_count = sum(1 for ts in history if timestamp - ts < timedelta(hours=24))
    if history and (timestamp - max(history)) < ALERT_COOLDOWN:
        return False, "Suppressed: another alert was dispatched for this account within the last hour", recent_count
    return True, "No recent alert for this account -- safe to dispatch", recent_count


def build_context(row, dispatched_history: dict) -> AlertContext:
    tags = tag_patterns(row)
    can_dispatch_by_history, reason, recent_count = check_account_history(
        row["account_id"], row["timestamp"], dispatched_history
    )
    # dispatch_safe is a plain Python decision: severity must be Red/Yellow
    # AND the account must be outside its cooldown window.
    dispatch_safe = row["severity"] in ("Red", "Yellow") and can_dispatch_by_history
    if row["severity"] == "Green":
        reason = "Severity is Green -- not dispatch-worthy"

    return AlertContext(
        txn_id=row["txn_id"],
        account_id=row["account_id"],
        account_name=row["account_name"],
        timestamp=row["timestamp"].isoformat(),
        amount=round(float(row["amount"]), 2),
        merchant_category=row["merchant_category"],
        severity=row["severity"],
        anomaly_score=round(float(row["anomaly_score"]), 3),
        tags=tags,
        recent_alerts_for_account=recent_count,
        dispatch_safe=dispatch_safe,
        dispatch_reason=reason,
    )


# --------------------------------------------------------------------------
# Layer B
# --------------------------------------------------------------------------
def _fallback_template(ctx: AlertContext) -> dict:
    urgency = {"Red": "high", "Yellow": "medium", "Green": "low"}[ctx.severity]
    escalation_required = ctx.severity == "Red"
    tag_str = ", ".join(ctx.tags).lower()

    return {
        "headline": f"{ctx.severity}-severity alert on {ctx.account_name}",
        "explanation": (
            f"A ${ctx.amount:,.2f} transaction on {ctx.account_name} "
            f"({ctx.merchant_category}) was flagged for: {tag_str}. "
            f"Anomaly score: {ctx.anomaly_score:.2f}."
        ),
        "recommended_action": (
            "Escalate to the fraud queue and contact the accountholder to confirm."
            if escalation_required
            else "Review the transaction and reach out to the accountholder if needed."
        ),
        "urgency": urgency,
        "escalation_required": escalation_required,
    }


def _call_llama(ctx: AlertContext) -> dict | None:
    try:
        import ollama
    except ImportError:
        return None

    payload = {
        "txn_id": ctx.txn_id, "account_name": ctx.account_name, "amount": ctx.amount,
        "merchant_category": ctx.merchant_category, "severity": ctx.severity,
        "anomaly_score": ctx.anomaly_score, "tags": ctx.tags,
        "recent_alerts_for_account": ctx.recent_alerts_for_account,
    }
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            format="json",  # asks the model to constrain output to valid JSON
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        text = response["message"]["content"]
        parsed = json.loads(text.strip())
        if any(k not in parsed for k in REQUIRED_KEYS):
            return None
        return {k: parsed[k] for k in REQUIRED_KEYS}
    except Exception:
        # Covers: Ollama not installed, the daemon not running, the model
        # not pulled yet, a connection error, or a malformed response --
        # any of these should fall back to the template, never crash the
        # pipeline.
        return None


def generate_alert(ctx: AlertContext) -> dict:
    narrative = _call_llama(ctx)
    source = "llm" if narrative is not None else "template"
    if narrative is None:
        narrative = _fallback_template(ctx)

    alert = {
        "txn_id": ctx.txn_id, "account_id": ctx.account_id, "account_name": ctx.account_name,
        "timestamp": ctx.timestamp, "amount": ctx.amount, "severity": ctx.severity,
        "anomaly_score": ctx.anomaly_score, "tags": ctx.tags,
        "dispatch_safe": ctx.dispatch_safe, "dispatch_reason": ctx.dispatch_reason,
        "generated_by": source,
        **narrative,
    }
    alert["dispatch_safe"] = ctx.dispatch_safe  # Layer A always wins, even if the LLM tried to add its own
    return alert
