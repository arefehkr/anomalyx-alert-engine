# AnomalyX Alert Engine

A small end-to-end fraud detection pipeline: an Isolation Forest flags
anomalous bank transactions, and a two-layer AI agent turns each flagged
transaction into a plain-English alert for a fraud analyst.

```
transactions.csv  →  features  →  Isolation Forest  →  severity
                                                            │
                                                            ▼
                                     Layer A: Python rules engine
                                     (tag the pattern, check account
                                      history, decide dispatch_safe)
                                                            │
                                                            ▼
                                     Layer B: local Llama (via Ollama)
                                     writes the alert (or a template,
                                     if Ollama isn't running)
                                                            │
                                                            ▼
                                        console + alerts_log.json
```

## What it does

1. **Detects anomalies.** Four simple features per transaction — how far
   the amount is from the account's rolling 30-day average, how many
   transactions the account made in the last hour, whether it happened
   overnight, and whether the merchant category is one the account
   normally uses — feed an Isolation Forest, which scores every
   transaction and buckets it into **Red / Yellow / Green**.

2. **Explains itself with a two-layer agent.**
   - **Layer A (Python, deterministic):** a small rules engine tags *why*
     a transaction was flagged (e.g. "Large Amount Deviation", "Rapid
     Transaction Burst"), checks whether this account already had a
     recent alert (so a burst of 4 transactions doesn't produce 4
     alerts), and decides `dispatch_safe` — **in Python, never by the
     LLM**.
   - **Layer B (a local Llama model via Ollama, or a template if Ollama
     isn't running):** takes that context and writes the actual alert —
     headline, explanation, recommended action, urgency. Everything runs
     on your machine — no API key, no cost, no data leaving your laptop.

## Setup

```bash
pip install -r requirements.txt

# Install Ollama (one-time, free): https://ollama.com/download
ollama pull llama3.2

python run_pipeline.py
```

Works even if Ollama isn't installed or running — Layer B just uses the
template instead. Run `python src/generate_data.py` any time to
regenerate the synthetic dataset.

```bash
python run_pipeline.py --max-alerts 3   # cap alerts (useful while testing)
pytest tests/ -v                        # 12 tests
```

## Results (this synthetic demo dataset)

| | |
|---|---|
| Recall | **100%** — every injected anomaly was flagged |
| Precision | **62.7%** — about 1 in 3 alerts is a false positive |

That trade-off is expected and honest: catching everything means also
catching some legitimately unusual (but not fraudulent) spending. A real
deployment would tune the Red/Yellow thresholds in `src/detector.py`
against a cost model (false alert vs. missed fraud) rather than chase 100%
on both.

## Project structure

```
run_pipeline.py       # the whole pipeline, start to finish
src/
  generate_data.py    # synthetic accounts + transactions with injected anomalies
  features.py          # 4 features: spend_ratio, txn_count_1h, is_night, merchant_mismatch
  detector.py           # Isolation Forest + Red/Yellow/Green thresholds
  agent.py               # Layer A (rules engine) + Layer B (Llama via Ollama / template)
  alerter.py               # console formatting, JSON log, burst dedup, eval
tests/test_pipeline.py       # 12 tests
```
