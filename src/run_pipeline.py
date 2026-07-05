"""
=====================================================================
 FILE: run_pipeline.py
=====================================================================
ROLE IN THE PIPELINE:
    This is the ORCHESTRATOR. It is the single entry point that ties
    together all three earlier stages in order: fetch -> detect ->
    narrate. This is the file GitHub Actions calls every week, and the
    file you run manually to test the whole thing end to end.

DEPENDS ON:
    fetch_data.py, detect_regime.py, generate_brief.py -- imports and
    calls all three in sequence. Contains no analysis logic of its own;
    its only job is sequencing and configuration.

FEEDS INTO:
    data/history.json (written by generate_brief.save_brief_to_history),
    which app.py reads to populate the dashboard.

WHY SERIES_CONFIG IS A LIST OF TUPLES:
    Each series needs different settings (which CSV file, which column
    holds the value, which detection mode applies). Rather than writing
    three near-identical blocks of code, one per series, this file loops
    over a config list and reuses the same three-stage call for each.
    Adding a fourth series later means adding one line to this list, not
    writing new pipeline code.

WHY PROJECT_ROOT IS COMPUTED EXPLICITLY (a real bug found during testing):
    Running this script from inside src/ vs. from the project root used
    to silently write data/ into the wrong place (a duplicate src/data/
    folder), because plain relative paths like "data/raw/..." resolve
    relative to wherever the terminal's current directory happens to
    be, not relative to the project itself. Computing PROJECT_ROOT from
    __file__ makes every path in this file correct regardless of which
    directory you're standing in when you run it.
=====================================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()  # loads GROQ_API_KEY from a local .env file, if one exists.
               # Harmless no-op in GitHub Actions, where the key instead
               # arrives as a real environment variable from a repo secret.

import pandas as pd
from fetch_data import fetch_all
from detect_regime import analyze_series
from generate_brief import generate_brief, save_brief_to_history

# Resolves to the actual project root, regardless of which directory
# this script is run from (e.g. `python run_pipeline.py` from inside
# src/, or `python src/run_pipeline.py` from the project root).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Each tuple: (csv filename, value column name, date column name, detection mode, penalty)
# mode="level" for administered/step series (Bank Rate, CPI)
# mode="returns" for market/random-walk-like series (GBP/USD)
SERIES_CONFIG = [
    ("bank_rate.csv", "bank_rate", "date", "level", 1.0),
    ("gbpusd.csv", "gbpusd", "date", "returns", 1.0),
    ("cpi.csv", "cpi_rate", "date", "level", 1.0),
]


def run():
    """
    INPUT:
        None -- all configuration comes from SERIES_CONFIG above.

    PROCESS:
        1. Call fetch_all() once, refreshing all three raw CSVs, saving
           into PROJECT_ROOT/data/raw (not a relative "data/raw").
        2. For each configured series:
             a. Read its CSV into a DataFrame, from PROJECT_ROOT/data/raw.
             b. Run analyze_series() (change-point detection + HMM).
             c. Run generate_brief() to narrate the result.
             d. Print the brief (visible in GitHub Actions logs, useful
                for debugging a scheduled run without needing to open
                the dashboard).
             e. Append the result to PROJECT_ROOT/data/history.json.

    OUTPUT:
        None returned. Side effects: refreshed CSVs in data/raw/, and
        a new entry per series appended to data/history.json.

    REASON THIS APPROACH:
        Looping over SERIES_CONFIG instead of hand-writing one block
        per series means the exact same, already-validated three-stage
        logic (fetch was already tested per-source; analyze_series was
        validated on both real Bank Rate and synthetic FX data; the
        brief format was tuned once in generate_brief.py) gets reused
        identically everywhere, so there's only one place to fix
        anything if a bug is found later.
    """
    print("Step 1: fetching latest data...")
    fetch_all(save_dir=os.path.join(PROJECT_ROOT, "data", "raw"))

    for filename, value_col, date_col, mode, pen in SERIES_CONFIG:
        print(f"\nStep 2: analyzing {value_col} (mode={mode})...")
        filepath = os.path.join(PROJECT_ROOT, "data", "raw", filename)
        df = pd.read_csv(filepath, parse_dates=[date_col])
        analysis = analyze_series(df, value_col=value_col, date_col=date_col, mode=mode, pen=pen)

        print(f"Step 3: generating brief for {value_col}...")
        brief = generate_brief(analysis)
        print(brief)

        history_path = os.path.join(PROJECT_ROOT, "data", "history.json")
        save_brief_to_history(analysis, brief, history_path=history_path)

    print("\nPipeline complete.")


if __name__ == "__main__":
    run()