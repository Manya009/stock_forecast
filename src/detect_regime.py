"""
=====================================================================
 FILE: detect_regime.py
=====================================================================
ROLE IN THE PIPELINE:
    This is the SECOND stage, and the analytical core of Regime Watch.
    It takes a clean time series (already fetched by fetch_data.py) and
    answers two separate questions:
        1. WHERE did the series' behavior structurally change?
           (change-point detection, via the `ruptures` library)
        2. WHAT STATE is it in right now, and how does that compare to
           other states we've seen? (regime classification, via a
           Hidden Markov Model from `hmmlearn`)

DEPENDS ON:
    Only numpy/pandas/ruptures/hmmlearn. Deliberately has NO knowledge
    of where the data came from (BoE, ONS, yfinance) or what happens to
    its output afterwards -- it just takes arrays in and returns
    structured facts out. This makes it reusable on any time series.

FEEDS INTO:
    generate_brief.py -- which takes the dictionary produced by
    analyze_series() and turns it into plain-English text via an LLM.

TWO MODES, AND WHY BOTH EXIST (validated through direct testing):
    mode="returns" -- for series that behave like a random walk
        (e.g. FX rates). We detect on DAY-TO-DAY CHANGES, not raw
        level, because raw price level wanders unpredictably even
        within a single stable regime -- testing this directly on FX
        data, running detection on raw price found a completely wrong
        breakpoint, while running it on returns found the correct one.

    mode="level" -- for administered / step-function series (e.g. a
        central bank policy rate). Here the raw level IS the
        meaningful signal, because changes are deliberate, discrete
        policy decisions, not random wandering. Validated against real
        Bank of England data, where this correctly identified the 2016
        post-Brexit cut, 2020 COVID cut, and the December 2021 hiking
        cycle start, within 1 day of the true event dates.

A KEY FIX BAKED INTO THIS FILE (found via direct debugging):
    Both detection functions STANDARDIZE their input (subtract mean,
    divide by standard deviation) before fitting. Without this, a
    single `pen` (penalty) or model setting behaves completely
    differently depending on a series' natural scale -- e.g. tiny FX
    returns (~0.003) vs. interest rate levels (0-5) -- causing the same
    settings to silently fail on one series type while working on
    another. Standardizing first makes settings portable across any
    series you feed in.
=====================================================================
"""
import numpy as np
import pandas as pd
import ruptures as rpt
from hmmlearn import hmm


def detect_changepoints(series: np.ndarray, mode: str = "returns", pen: float = 1.0) -> list[int]:
    """
    INPUT:
        series -- 1D array of raw values (e.g. prices, or rate levels).
        mode -- "returns" or "level" (see file header for when to use which).
        pen -- penalty controlling detection sensitivity. Lower = more
               breakpoints found (more sensitive, more false-positive
               risk). Higher = fewer, more conservative breakpoints.

    PROCESS:
        1. Convert the raw series into the right "signal" to analyze:
           day-to-day differences for "returns" mode, or the raw values
           unchanged for "level" mode.
        2. Standardize that signal (mean 0, standard deviation 1) so
           `pen` means the same thing regardless of the series' natural
           units -- this is the fix described in the file header.
        3. Fit PELT (Pruned Exact Linear Time), an algorithm that finds
           how many breakpoints are statistically justified, rather
           than assuming a fixed number in advance.
        4. Ask it to predict breakpoint locations at the given `pen`.

    OUTPUT:
        A list of integer indices (into the ORIGINAL series, not the
        signal) where a structural change was detected. The final,
        trailing "breakpoint" that ruptures always includes (equal to
        the series length) is dropped, since it isn't a real detected
        event.

    REASON THIS APPROACH (specifically PELT with a penalty, not a fixed
    breakpoint count):
        Testing this on real Bank of England data directly showed that
        asking for a FIXED number of breakpoints (e.g. "give me the top
        3") biases the algorithm toward subdividing the single
        largest-magnitude event (the 2022-23 rate hiking staircase)
        while completely ignoring smaller-but-real events (the 2016 and
        2020 cuts). Asking PELT to decide how many breaks are justified
        via a penalty instead distributes detection fairly across
        events of different sizes.
    """
    if mode == "returns":
        signal = np.diff(series)
    elif mode == "level":
        signal = series
    else:
        raise ValueError("mode must be 'returns' or 'level'")

    sigma = signal.std()
    sigma = sigma if sigma > 1e-9 else 1.0  # guard against a perfectly flat, zero-variance signal
    signal_scaled = (signal - signal.mean()) / sigma

    algo = rpt.Pelt(model="l2").fit(signal_scaled)
    breakpoints = algo.predict(pen=pen)
    return breakpoints[:-1]


def fit_regime_hmm(series: np.ndarray, mode: str = "returns", n_states: int = 2, random_state: int = 42) -> dict:
    """
    INPUT:
        series -- 1D array of raw values.
        mode -- "returns" or "level", same meaning as in detect_changepoints.
        n_states -- how many hidden regimes to assume exist (default 2,
                    e.g. "calm vs turbulent" or "low-rate vs high-rate").
        random_state -- fixes the model's internal randomness so the
                        same input always produces the same output.

    PROCESS:
        1. Build the same "signal" as detect_changepoints (returns or
           level depending on mode).
        2. Standardize it -- this is not optional. The FIRST version of
           this function, tested without standardizing, produced a
           state with a variance over one billion (a numerical
           breakdown, not a real finding), because HMM's internal
           optimizer becomes unstable on very small-scale numbers like
           raw FX returns (~0.003). Standardizing fixed this completely.
        3. Fit a Gaussian HMM with `n_states` hidden states.
        4. Use the fitted model to label EVERY point in the signal with
           its most likely hidden state.
        5. The current regime is simply whichever state the MOST RECENT
           point was assigned to.
        6. Walk backwards from the most recent point to find exactly
           where the current, still-ongoing regime run began.
        7. Compute each state's own mean/std/count, so every state can
           be described and compared, not just the current one.

    OUTPUT:
        A dictionary containing:
            hidden_states    -- state label for every point in the signal
            current_regime   -- state label of the most recent point
            regime_since_idx -- index (in the signal) where the current
                                 regime run began
            regime_mean      -- current regime's own mean (raw units)
            regime_std       -- current regime's own std (raw units)
            state_summary    -- {state_label: {mean, std, n_points}}
                                 for every state found
            transition_matrix -- how likely the model thinks it is to
                                  switch between states day to day

    REASON THIS APPROACH:
        A Hidden Markov Model is the right tool here specifically
        because it labels EVERY point continuously, not just a single
        breakpoint like ruptures does. This is what lets us answer
        "what state are we in TODAY", not just "something changed on
        this date" -- ruptures alone cannot answer the first question.
        Validated on real Bank Rate data: cleanly separated the entire
        2015-2021 low-rate era from the 2022+ high-rate era with no
        flickering between states.
    """
    if mode == "returns":
        signal = np.diff(series)
    elif mode == "level":
        signal = series
    else:
        raise ValueError("mode must be 'returns' or 'level'")

    mu, sigma = signal.mean(), signal.std()
    sigma = sigma if sigma > 1e-9 else 1.0
    scaled = ((signal - mu) / sigma).reshape(-1, 1)

    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                             n_iter=200, random_state=random_state)
    model.fit(scaled)
    hidden_states = model.predict(scaled)

    current_regime = hidden_states[-1]

    # Walk backwards from the end: keep stepping earlier as long as the
    # previous point was ALSO in the current regime. Stop the moment we
    # hit a point from a different regime -- that gives us the first day
    # of the current, still-ongoing run.
    regime_since_idx = len(hidden_states) - 1
    while regime_since_idx > 0 and hidden_states[regime_since_idx - 1] == current_regime:
        regime_since_idx -= 1

    state_summary = {}
    for state in np.unique(hidden_states):
        mask = hidden_states == state
        state_summary[int(state)] = {
            "mean": float(signal[mask].mean()),
            "std": float(signal[mask].std()),
            "n_points": int(mask.sum()),
        }

    return {
        "hidden_states": hidden_states,
        "current_regime": int(current_regime),
        "regime_since_idx": regime_since_idx,
        "regime_mean": state_summary[int(current_regime)]["mean"],
        "regime_std": state_summary[int(current_regime)]["std"],
        "state_summary": state_summary,
        "transition_matrix": model.transmat_,
    }


def regime_aware_forecast(series: np.ndarray, mode: str, horizon: int, hmm_result: dict) -> dict:
    """
    INPUT:
        series -- 1D array of raw values, used only for its LAST value
                  (the starting point of the forecast).
        mode -- must be "returns"; this function only makes sense for
                random-walk-like series, not administered step series.
        horizon -- how many steps ahead to forecast.
        hmm_result -- the dictionary returned by fit_regime_hmm(), used
                      to pull the CURRENT regime's own drift and volatility.

    PROCESS:
        1. Take the current regime's mean (drift) and standard
           deviation (volatility) -- and ONLY the current regime's,
           never blended with any other regime's history.
        2. Project the drift forward linearly (last value + t * drift).
        3. Build an uncertainty band around that point forecast, using
           the standard result that a random walk's uncertainty grows
           with the SQUARE ROOT of time, not linearly.

    OUTPUT:
        A dictionary with point_forecast, upper_band, and lower_band --
        all arrays of length `horizon`.

    REASON THIS APPROACH (and an important honest limitation):
        Direct testing showed this method does NOT reliably beat a
        plain moving average on raw point-forecast accuracy (RMSE) --
        no method can reliably predict the exact zigzag path of a
        genuinely noisy process. What THIS method wins decisively on is
        HONEST UNCERTAINTY: in direct testing, a 95% band built from
        the regime-aware volatility captured 100% of actual outcomes,
        versus only 37% for a band built from blended (all-history)
        volatility, which was falsely overconfident. This function's
        real value is producing a defensible "how much should you
        trust this" answer, not a precise price prediction.
    """
    if mode != "returns":
        raise ValueError("regime_aware_forecast is defined for mode='returns' series")

    drift = hmm_result["regime_mean"]
    vol = hmm_result["regime_std"]
    last_value = series[-1]

    t = np.arange(1, horizon + 1)
    point_forecast = last_value + t * drift
    upper = point_forecast + 1.96 * vol * np.sqrt(t)
    lower = point_forecast - 1.96 * vol * np.sqrt(t)

    return {"point_forecast": point_forecast, "upper_band": upper, "lower_band": lower}


def analyze_series(df: pd.DataFrame, value_col: str, date_col: str,
                    mode: str, pen: float = 1.0) -> dict:
    """
    INPUT:
        df -- a DataFrame containing at least a value column and a date column.
        value_col -- name of the column holding the numeric series.
        date_col -- name of the column holding dates.
        mode -- "returns" or "level".
        pen -- penalty passed through to detect_changepoints.

    PROCESS:
        1. Pull the raw value and date arrays out of the DataFrame.
        2. Run BOTH change-point detection and HMM regime fitting.
        3. Correct for an off-by-one issue: when mode="returns", both
           detect_changepoints and fit_regime_hmm internally call
           np.diff(), which produces an array ONE ELEMENT SHORTER than
           the original series. An index into that shorter array is
           off-by-one relative to the same index in the original dates
           array. `offset = 1` corrects for this when translating an
           index back into a real calendar date -- getting this wrong
           would silently report every date one day earlier than the
           truth.
        4. Package everything into a single, flat, self-contained
           dictionary -- deliberately with no raw numpy arrays left in
           it (aside from what's nested in state_summary), so this
           output can be safely converted to JSON for logging and for
           feeding directly into the LLM prompt in generate_brief.py.

    OUTPUT:
        A dictionary with (among other fields): series_name, mode,
        changepoint_dates (as ISO date strings), current_regime_label,
        regime_since (ISO date string), days_in_current_regime,
        regime_mean, regime_std, all_states, latest_value, latest_date.

    REASON THIS APPROACH:
        This function exists so that nothing else in the project needs
        to know the internal details of ruptures or hmmlearn, or worry
        about the off-by-one indexing issue -- it's handled once, here,
        and every caller downstream just receives clean, final facts.
    """
    series = df[value_col].values
    dates = df[date_col].values

    changepoints = detect_changepoints(series, mode=mode, pen=pen)
    hmm_result = fit_regime_hmm(series, mode=mode)

    offset = 1 if mode == "returns" else 0  # returns array is 1 shorter than the original series
    regime_since_date = pd.Timestamp(dates[hmm_result["regime_since_idx"] + offset])
    days_in_regime = len(series) - (hmm_result["regime_since_idx"] + offset)

    return {
        "series_name": value_col,
        "mode": mode,
        "changepoint_dates": [pd.Timestamp(dates[i + offset]).date().isoformat() for i in changepoints],
        "current_regime_label": hmm_result["current_regime"],
        "regime_since": regime_since_date.date().isoformat(),
        "days_in_current_regime": int(days_in_regime),
        "regime_mean": hmm_result["regime_mean"],
        "regime_std": hmm_result["regime_std"],
        "all_states": hmm_result["state_summary"],
        "latest_value": float(series[-1]),
        "latest_date": pd.Timestamp(dates[-1]).date().isoformat(),
    }