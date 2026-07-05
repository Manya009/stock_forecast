# Regime Watch

An AI-narrated trend and regime detection dashboard for UK economic and financial data,
built to be understandable by a non-technical audience while remaining fully rigorous
underneath.

**Live idea:** most forecasting tools assume the recent past is a reliable guide to the
near future. That assumption breaks exactly when it matters most — right as conditions
structurally change. Regime Watch instead detects *that* a change happened and
characterizes *what state* things are in now, and communicates it in plain English.

## What it actually does

1. Pulls three real UK data series every week:
   - **Bank of England Bank Rate** (interest rates)
   - **ONS CPI inflation**
   - **GBP/USD** exchange rate
2. Runs statistical **change-point detection** (`ruptures`) and **Hidden Markov Model**
   regime classification (`hmmlearn`) on each series.
3. Sends the results to an LLM (Groq, Llama 3.3 70B) with a strict plain-English prompt
   — no jargon, no raw regime labels, no unexplained statistics.
4. Displays everything on an interactive Streamlit dashboard: a one-sentence plain-
   language headline, an interactive Plotly chart with regime shading, the written
   brief, and a collapsed "technical details" section for anyone who wants the raw
   numbers.
5. Repeats automatically every week via GitHub Actions — no manual re-running required.

## Why regime detection instead of price prediction

This was deliberately validated, not just assumed. Direct testing (documented in
`notebooks/`) showed:

- A moving average, ARIMA, and Prophet were all tested against a real, dramatic regime
  change (UK Bank Rate, 2015–2024) and each failed to anticipate it in a different way.
- Change-point detection (`ruptures`), once correctly configured (detecting on returns
  vs. raw level depending on series type, and using a penalty-based method rather than a
  fixed breakpoint count), correctly identified the 2016 Brexit-era cut, the 2020 COVID
  cut, and the December 2021 hiking cycle start — each within a day or two of the real
  historical event.
- A "regime-aware" forecast did **not** beat a naive moving average on raw point-forecast
  accuracy (RMSE) — but its uncertainty band captured **100%** of actual outcomes in
  testing, versus only **37%** for a naive band built by blending regimes together. The
  real value of this approach is honestly calibrated uncertainty, not a falsely precise
  prediction.

## Project structure

```
regime_watch/
├── .github/workflows/refresh.yml   # weekly automation: fetch -> detect -> narrate -> commit
├── data/
│   ├── raw/                         # bank_rate.csv, cpi.csv, gbpusd.csv
│   └── history.json                 # every brief ever generated, with its underlying stats
├── notebooks/                       # exploratory work: synthetic-data testing, real-data validation
├── src/
│   ├── fetch_data.py                 # BoE / ONS / yfinance data pulls
│   ├── detect_regime.py              # change-point detection + HMM regime labeling
│   ├── generate_brief.py             # Groq API narration, plain-English system prompt
│   └── run_pipeline.py               # orchestrator: ties fetch -> detect -> narrate together
├── app.py                            # Streamlit dashboard
├── requirements.txt
└── .env                              # holds GROQ_API_KEY locally (never committed)
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

(Free key from [console.groq.com](https://console.groq.com))

Run the full pipeline once (fetches real data, detects regimes, generates briefs):

```bash
python src/run_pipeline.py
```

Then view the dashboard:

```bash
streamlit run app.py
```

**Important:** run both commands from the project root, not from inside `src/` —
relative paths are resolved from wherever the terminal's current directory is.

## Deploying

1. Push this repo to GitHub (`.env`, `.venv`/`myenv`, and `__pycache__` are already
   excluded via `.gitignore`)
2. Add `GROQ_API_KEY` as a repository secret: **Settings → Secrets and variables →
   Actions → New repository secret**
3. Connect the repo to [Streamlit Community Cloud](https://share.streamlit.io), deploy
   `app.py`, for a free public URL
4. The GitHub Action refreshes data and regenerates briefs automatically every Monday —
   trigger it manually any time from the **Actions** tab (`workflow_dispatch`)

## Roadmap

- [ ] **Accuracy tracker** — log whether each regime's implied volatility actually held
      up over the following weeks, building an honest, checkable track record over time
- [ ] Additional series (FTSE 100, UK gilt yields, unemployment)
- [ ] Model-selection for HMM state count (currently fixed at 2 states) using BIC
