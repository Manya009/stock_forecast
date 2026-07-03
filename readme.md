# Regime Watch — AI-Narrated Trend & Regime Detection Dashboard

**A free, ongoing portfolio project combining statistical time-series analysis with LLM narration.**

---

## 1. Project Overview

**What it does:** Pulls UK/global macro and market time series, runs statistical regime/change-point detection on them, and uses an LLM to turn the raw statistical output into a short, plain-English "brief" — like a junior economist's weekly note. Published as a public dashboard and refreshed automatically.

**Why it's a good portfolio piece for you:**
- Extends your AI for Finance coursework (regime change models, structural break models, vectorized backtesting) into a live, working system
- Extends your CypherSOL forensic analytics story (turning technical findings into plain-language output for non-technical audiences)
- Recurring, not a one-off — gives you a natural reason to post on LinkedIn weekly without running out of content
- Has a built-in "proof" mechanism (tracking whether past regime calls aged well), which is rare and memorable

**Target audience for the LinkedIn angle:** recruiters and DS/MLE hiring managers who want to see you can (a) build real statistical models, not just prompt an LLM, and (b) ship and maintain something in production.

---

## 2. Architecture

```
[Data Sources]              [Analysis Layer]           [LLM Layer]           [Presentation]
BoE API      ─┐                                                                     
ONS API      ─┼──► raw CSV/DF ──► ruptures (change-point) ──► Groq API ──► Streamlit app
yfinance     ─┘                   hmmlearn (regime HMM)        (brief gen)     (public URL)
                                                                    │
                                                                    ▼
                                                          history.json (git-committed)
                                                          — powers the accuracy tracker
                                                                    │
                                                                    ▼
                                                        GitHub Actions (weekly cron)
                                                        re-runs the whole pipeline
```

---

## 3. Tech Stack (100% free)

| Layer | Tool | Cost |
|---|---|---|
| Data | Bank of England API, ONS API, `yfinance` | Free |
| Change-point detection | `ruptures` | Free, open-source |
| Regime detection | `hmmlearn` | Free, open-source |
| Forecasting baseline (optional, v2) | `statsmodels` / `prophet` | Free, open-source |
| LLM narration | Groq API (Llama 3.3 70B) | Free tier |
| Automation | GitHub Actions | Free (2,000 min/mo) |
| Hosting | Streamlit Community Cloud | Free |
| Storage | Git-committed CSV/JSON | Free |

---

## 4. Repo Structure

```
regime-watch/
├── .github/workflows/refresh.yml     # weekly cron job
├── data/
│   ├── raw/                          # pulled data, git-committed
│   └── history.json                  # logged regime calls + outcomes
├── src/
│   ├── fetch_data.py                 # pulls BoE/ONS/yfinance data
│   ├── detect_regime.py              # ruptures + hmmlearn logic
│   ├── generate_brief.py             # Groq API call, builds the narrative
│   └── accuracy_tracker.py           # v3: checks past calls vs reality
├── app.py                            # Streamlit dashboard
├── requirements.txt
├── .env.example
└── README.md
```

---

## 5. Step-by-Step Build Plan

### Phase 0 — Setup (30 min)
1. Create GitHub repo `regime-watch`
2. Set up a Python venv, `pip install ruptures hmmlearn yfinance streamlit groq requests pandas`
3. Get a free Groq API key, store in `.env` (never commit it — add `.env` to `.gitignore`)

### Phase 1 — Data layer (half day)
1. Write `fetch_data.py`:
   - Pull GBP/USD daily close via `yfinance` (`yf.download("GBPUSD=X")`)
   - Pull UK Bank Rate history from the BoE API
   - Pull CPI inflation from the ONS API
2. Save each as a clean CSV in `data/raw/`
3. **Milestone:** running `python src/fetch_data.py` produces 3 up-to-date CSVs

### Phase 2 — Analysis layer (1 day — this is your differentiator, don't rush it)
1. Write `detect_regime.py`:
   - Use `ruptures.Pelt` or `ruptures.Binseg` to find structural breaks in each series
   - Fit a 2-3 state Gaussian HMM with `hmmlearn` to label "regimes" (e.g. low-vol/high-vol, tightening/loosening)
2. Output a simple structured result per series, e.g.:
   ```json
   {"series": "GBPUSD", "last_break": "2026-06-18", "current_regime": "high_volatility", "regime_since": "2026-06-18"}
   ```
3. **Milestone:** the script correctly flags a known historical break (e.g. a BoE rate decision date) as a test sanity check

### Phase 3 — LLM narration layer (half day)
1. Write `generate_brief.py`:
   - Feed the structured JSON from Phase 2 into a Groq API call
   - System prompt: instruct the model to write a 150-200 word plain-English brief, no fluff, no hedging clichés, specific numbers included
   - Store the output brief + the raw stats together in `data/history.json`, timestamped
2. **Milestone:** running the script end-to-end produces a readable brief you'd be comfortable posting publicly

### Phase 4 — Streamlit UI (1 day)
1. `app.py`:
   - Line chart per series with break-points and regime shading overlaid (Plotly or Altair inside Streamlit)
   - The latest LLM brief displayed prominently at the top
   - A simple archive view of past briefs
2. **Milestone:** `streamlit run app.py` looks presentable locally

### Phase 5 — Automation (half day)
1. `.github/workflows/refresh.yml`: cron trigger (e.g. every Monday), runs `fetch_data.py` → `detect_regime.py` → `generate_brief.py`, commits updated data back to the repo
2. Store the Groq API key as a GitHub Actions secret
3. **Milestone:** trigger the workflow manually once and confirm it commits fresh data

### Phase 6 — Deploy (1-2 hours)
1. Push repo to GitHub, connect to Streamlit Community Cloud, deploy `app.py`
2. You now have a public URL — this is what goes in your LinkedIn post

### Phase 7 — Launch content
1. Write a LinkedIn post: what it does, why you built it (tie to your Manchester coursework + CypherSOL background), link to the live app + repo
2. Plan a recurring weekly post: "This week's AI-generated regime brief" + one line of your own commentary

---

## 6. v2 / v3 Roadmap (keeps it "ongoing")

- **v2:** Add more series (FTSE 100, UK gilt yields, unemployment), add a simple forecast baseline for context
- **v3 (the standout feature):** Accuracy tracker — each week, check whether the regime called 4 weeks ago actually held, log a running "hit rate." This is rare in public LLM demo projects and gives you something concrete and quantifiable to talk about in interviews

---

## 7. Suggested Timeline

| Week | Focus |
|---|---|
| 1 | Phases 0-2 (data + analysis layer) |
| 2 | Phases 3-4 (LLM narration + UI) |
| 3 | Phases 5-6 (automation + deploy), first LinkedIn post |
| Ongoing | Weekly briefs, v2/v3 features added incrementally |

---

## 8. Next Step

Ready-to-run starter code for Phase 1 and 2 (data pull + change-point detection) so you can get the first milestone working today.