# 📊 Regime Watch

**An AI-narrated dashboard that detects when UK financial and economic trends genuinely change, not just when the numbers go up or down.**

[Live Demo](#) · [How it works](#how-it-works) · [Validated findings](#validated-findings)

---

## The problem

Most forecasting tools assume the recent past is a reliable guide to the near future. That assumption works fine, until it doesn't, and it always breaks at the worst possible moment: right when conditions genuinely change.

**Regime Watch** takes a different approach. Instead of trying to predict the next number, it detects *when* a market or economic series has entered a structurally different phase (a "regime"), characterizes what that phase actually looks like, and explains it in plain English, automatically, every week.

## What it tracks

| Series | Source |
|---|---|
| UK Bank Rate | Bank of England |
| UK CPI Inflation | Office for National Statistics |
| GBP/USD | Yahoo Finance |

## How it works

```
Fetch real data  →  Detect structural change-points  →  Classify the current regime  →  Narrate in plain English
   (weekly)          (ruptures, PELT algorithm)         (Hidden Markov Model)              (Llama 3.3 70B via Groq)
```

1. **Change-point detection** finds the exact moments a series' behavior structurally shifted, using a penalty-based method rather than a fixed count, so it doesn't over-focus on the single largest event and miss smaller-but-real ones.
2. **Regime classification** (Hidden Markov Model) labels every day with which "state" the series was in, and tracks how long the current state has persisted.
3. **An LLM narrates the result** in plain English, explicitly instructed to avoid statistical jargon, raw regime labels, and invented predictions.
4. **A GitHub Action re-runs the whole pipeline weekly**, so the dashboard stays current with no manual work.

## Validated findings

This wasn't just built and shipped; the core method was stress-tested against real, independently-checkable historical events before being trusted.

- **Detected the December 2021 UK rate hiking cycle within 1 day of the actual first hike**, along with the 2020 COVID rate cut and the 2016 post-Brexit cut, using nothing but statistical detection on public data, no manual date-hinting involved.
- **A "regime-aware" forecast's uncertainty band captured 100% of actual outcomes in testing, versus 37% for a naive approach that blends different regimes together.** The naive approach's narrower, more confident-looking band was actually far less trustworthy. This project optimizes for honest uncertainty over false precision.
- Standard tools (moving averages, ARIMA, Prophet) were each tested against the same real regime change and each failed to anticipate it in a distinct, informative way, documented in `/notebooks`.

## Why this design

The LLM is deliberately kept as a **narrator, not a detector**. All statistical work happens first, using validated, testable methods; the LLM only ever translates a finished, structured result into plain language. It never sees raw data and is never asked to "figure out" a regime itself, which would invite hallucinated statistics.

## Tech stack

Python · `ruptures` · `hmmlearn` · `statsmodels` · Groq (Llama 3.3 70B) · Streamlit · Plotly · GitHub Actions

Entirely free to run: public data sources, free-tier LLM inference, free compute (GitHub Actions), free hosting (Streamlit Community Cloud).

## Running it yourself

```bash
git clone https://github.com/Manya009/stock_forecast
cd stock_forecast
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env    # free key at console.groq.com

python src/run_pipeline.py    # fetch real data, detect regimes, generate briefs
streamlit run app.py          # view the dashboard
```

## Roadmap

- [ ] Accuracy tracker: log whether each regime's implied volatility actually held up over the following weeks, building a checkable track record
- [ ] Additional series (FTSE 100, UK gilt yields, unemployment)
- [ ] Automatic HMM state-count selection (currently fixed at 2 states)

---

Built by [Manish Patil](https://linkedin.com/in/manish-patil-1303aa215) · [GitHub](https://github.com/Manya009)
