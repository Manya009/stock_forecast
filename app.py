"""
=====================================================================
 FILE: app.py
=====================================================================
ROLE IN THE PIPELINE:
    The DISPLAY layer -- reads data/history.json (narrated briefs) and
    data/raw/*.csv (re-analyzed only for charting), and renders them.
    Does no fetching, detection, or LLM calls of its own.

DESIGN PRINCIPLE FOR THIS VERSION:
    A first-time, non-technical visitor should understand the headline
    finding in about 3 seconds, without knowing what "regime 1" or a
    standard deviation is. Concretely:
        - Raw HMM state labels (0, 1, ...) are NEVER shown directly --
          they're translated into a plain description ("Calmer than
          usual" / "More volatile than usual" / "Higher than usual" /
          "Lower than usual") by comparing the current state's stats
          against the other state(s).
        - The plain-language headline and the AI brief are the most
          visually prominent things on the page.
        - All raw numbers (means, std devs, exact regime labels) are
          still available, but tucked into a collapsed "technical
          details" section for anyone who wants to dig deeper.

RUN WITH:
    streamlit run app.py   (from the PROJECT ROOT, not from inside src/)
=====================================================================
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from detect_regime import fit_regime_hmm

st.set_page_config(page_title="Regime Watch", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    h1 { font-weight: 700; letter-spacing: -0.02em; }
    .headline-card {
        padding: 24px 28px;
        border-radius: 12px;
        margin-bottom: 18px;
        color: #1a202c;
    }
    .headline-card h2 { margin: 0 0 6px 0; font-size: 1.5rem; color: #1a202c; }
    .headline-card p { margin: 0; opacity: 0.8; font-size: 1rem; color: #1a202c; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Regime Watch")
st.caption("AI-narrated trend and regime detection for UK economic & financial data")

SERIES_FILES = {
    "bank_rate": {"filename": "bank_rate.csv", "date_col": "date", "mode": "level", "label": "UK Bank Rate (%)"},
    "gbpusd": {"filename": "gbpusd.csv", "date_col": "date", "mode": "returns", "label": "GBP/USD"},
    "cpi_rate": {"filename": "cpi.csv", "date_col": "date", "mode": "level", "label": "UK CPI Inflation Rate (%)"},
}

# (background color, accent color, emoji) per plain-language category
STYLE_TURBULENT = ("#fff5f5", "#c53030", "⚡")
STYLE_CALM = ("#f0f7ff", "#2b6cb0", "🌊")
STYLE_HIGH = ("#fff5f5", "#c53030", "📈")
STYLE_LOW = ("#f0f7ff", "#2b6cb0", "📉")


def describe_regime(analysis: dict) -> dict:
    """
    INPUT:
        analysis -- the dict from history.json for one series (same
        shape produced by detect_regime.analyze_series).

    PROCESS:
        1. Look at the current regime's stats vs. every OTHER regime's
           stats already stored in analysis["all_states"].
        2. For mode="returns" (market/random-walk series): compare
           VOLATILITY (std dev). Higher than the others -> "more
           volatile than usual". Lower -> "calmer than usual".
        3. For mode="level" (administered series like Bank Rate):
           compare the MEAN level. Higher -> "higher than usual".
           Lower -> "lower than usual".
        4. Pick a headline sentence, an emoji, and a background/accent
           color pair to match.

    OUTPUT:
        A dict: {headline, subtext, bg_color, accent_color, emoji}

    REASON THIS APPROACH:
        This is the plain-language translation layer. The raw HMM
        state label (an arbitrary integer like 0 or 1) is NEVER shown
        to the visitor directly -- it's compared numerically against
        the other state(s) here, and only the resulting plain
        description reaches the page. A visitor never needs to know
        what a "state" or "regime label" even is.
    """
    current_label = analysis["current_regime_label"]
    all_states = analysis["all_states"]
    current_stats = all_states[str(current_label)] if str(current_label) in all_states else all_states[current_label]

    other_stats = [v for k, v in all_states.items() if str(k) != str(current_label)]
    mode = analysis["mode"]
    days = analysis["days_in_current_regime"]
    since = analysis["regime_since"]

    if mode == "returns":
        avg_other_std = np.mean([s["std"] for s in other_stats]) if other_stats else current_stats["std"]
        if current_stats["std"] > avg_other_std:
            bg, accent, emoji = STYLE_TURBULENT
            headline = "More volatile than usual"
        else:
            bg, accent, emoji = STYLE_CALM
            headline = "Calmer than usual"
        subtext = f"This pattern has held for {days} days, since {since}."
    else:  # mode == "level"
        avg_other_mean = np.mean([s["mean"] for s in other_stats]) if other_stats else current_stats["mean"]
        if current_stats["mean"] > avg_other_mean:
            bg, accent, emoji = STYLE_HIGH
            headline = "Higher than its historical average"
        else:
            bg, accent, emoji = STYLE_LOW
            headline = "Lower than its historical average"
        subtext = f"This level has held for {days} days, since {since}."

    return {"headline": headline, "subtext": subtext, "bg_color": bg, "accent_color": accent, "emoji": emoji}


@st.cache_data
def load_history(path: str = "data/history.json") -> list:
    """
    INPUT:  path to the JSON history log.
    PROCESS: reads and parses; returns [] if the file doesn't exist yet.
    OUTPUT: list of brief entries.
    REASON: cached so Streamlit doesn't re-read the file on every click.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


@st.cache_data
def build_regime_chart(series_name: str) -> go.Figure:
    """
    INPUT:  series_name -- a key in SERIES_FILES.
    PROCESS: loads the raw CSV, re-fits the HMM (needed for the full
             per-day state sequence, which history.json doesn't store),
             draws the line plus background shading per regime run.
    OUTPUT: a Plotly Figure.
    REASON: Plotly gives free, native zoom/pan/hover interactivity in
            the browser -- chosen over embedding Power BI, which would
            need a paid Power BI Embedded/Azure setup to show reports
            publicly, breaking this project's zero-cost goal.
    """
    config = SERIES_FILES[series_name]
    df = pd.read_csv(f"data/raw/{config['filename']}", parse_dates=[config["date_col"]])
    series = df[series_name].values
    dates = df[config["date_col"]].values

    hmm_result = fit_regime_hmm(series, mode=config["mode"])
    hidden_states = hmm_result["hidden_states"]

    offset = 1 if config["mode"] == "returns" else 0
    plot_dates = dates[offset:]
    plot_values = series[offset:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_dates, y=plot_values, mode="lines",
        line=dict(color="#2b6cb0", width=1.5),
        name=config["label"],
        hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra></extra>",
    ))

    state_changes = np.where(np.diff(hidden_states) != 0)[0]
    segment_starts = np.concatenate([[0], state_changes + 1])
    segment_ends = np.concatenate([state_changes + 1, [len(hidden_states) - 1]])

    shade_colors = {0: "rgba(43,108,176,0.10)", 1: "rgba(229,62,62,0.10)"}
    for start, end in zip(segment_starts, segment_ends):
        state = int(hidden_states[start])
        fig.add_vrect(
            x0=plot_dates[start], x1=plot_dates[end],
            fillcolor=shade_colors.get(state, "rgba(150,150,150,0.10)"),
            line_width=0,
        )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


history = load_history()

if not history:
    st.warning(
        "No history found yet. Run `python src/run_pipeline.py` "
        "(from the project root) to populate data/history.json."
    )
else:
    latest_by_series = {}
    for entry in history:
        latest_by_series[entry["series_name"]] = entry

    tabs = st.tabs([SERIES_FILES.get(name, {}).get("label", name) for name in latest_by_series.keys()])

    for tab, (series_name, entry) in zip(tabs, latest_by_series.items()):
        with tab:
            analysis = entry["analysis"]
            desc = describe_regime(analysis)

            # --- The headline card: this is the ONE thing a first-time,
            # non-technical visitor needs to read. Everything else on
            # the page is supporting detail for anyone who wants more.
            st.markdown(f"""
            <div class="headline-card" style="background-color:{desc['bg_color']};
                 border-left: 5px solid {desc['accent_color']};">
                <h2>{desc['emoji']} {desc['headline']}</h2>
                <p>{desc['subtext']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(build_regime_chart(series_name), use_container_width=True)
            st.caption("🔵 and 🔴 shading show different behavioral periods. Zoom, pan, and hover for exact values.")

            st.subheader("The full story, in plain English")
            st.write(entry["brief"])
            st.caption(f"Last updated {entry['generated_at'][:10]}")

            with st.expander("🔍 Technical details (for the curious)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Regime label (internal)", analysis["current_regime_label"])
                    st.metric("Regime mean", f"{analysis['regime_mean']:.5f}")
                with col2:
                    st.metric("Days in current regime", analysis["days_in_current_regime"])
                    st.metric("Regime volatility (std)", f"{analysis['regime_std']:.5f}")

                st.write("**All detected historical states:**")
                st.json(analysis["all_states"])

                st.write("**Detected structural change-point dates:**")
                st.write(", ".join(analysis["changepoint_dates"]) or "None detected")

            with st.expander("📜 Brief history for this series"):
                series_history = [e for e in history if e["series_name"] == series_name]
                for e in reversed(series_history[-10:]):
                    st.markdown(f"**{e['generated_at'][:10]}**")
                    st.write(e["brief"])
                    st.divider()

st.divider()
st.subheader("What is this, exactly?")
st.markdown("""
Most financial charts just show you a line going up and down and leave you to guess
whether anything important happened. **Regime Watch** does something different: it
watches three UK economic indicators — the Bank of England's interest rate, UK
inflation, and the pound's exchange rate against the dollar — and automatically detects
when their *behavior* genuinely changes, not just when the number goes up or down.

**How to read this page:**
1. **Start with the colored box at the top of each tab.** That one sentence is the
   headline: is this series currently calmer or more turbulent than usual, higher or
   lower than its typical level?
2. **Look at the chart underneath it.** The colored shading in the background marks out
   different stretches of time where the series was behaving differently — you can
   zoom in, hover over any point, and explore the history yourself.
3. **Read "The full story, in plain English"** for a short written explanation of what's
   happening and how it compares to the past.
4. **Curious about the actual math?** Open "Technical details" underneath — that's where
   the real statistics live, for anyone who wants to dig deeper.

**Why this matters:** most simple forecasting tools assume the recent past is a good
guide to the near future — which works fine until it suddenly doesn't. This project is
built to catch exactly that moment: when something structurally changes, rather than
just reacting to it after the fact. Everything updates automatically every week, using
real UK government and market data.
""")
st.caption("Built with Python, statistical regime detection (change-point detection + Hidden Markov Models), and an LLM for plain-English narration.")