"""
=====================================================================
 FILE: generate_brief.py
=====================================================================
ROLE IN THE PIPELINE:
    This is the THIRD stage. It takes the structured dictionary
    produced by detect_regime.analyze_series() -- pure numbers and
    dates -- and turns it into a short, plain-English paragraph a
    human can actually read, using the Groq API (free tier, Llama
    3.3 70B).

DEPENDS ON:
    detect_regime.py's output shape (specifically the dictionary keys
    produced by analyze_series). Does not depend on ruptures or
    hmmlearn directly -- it never sees raw arrays, only the finished
    summary. This separation matters: the LLM's only job is
    TRANSLATION, not detection or analysis. It is never asked to look
    at raw numbers and "figure out" the regime itself -- that would
    invite hallucinated statistics. It only narrates facts it's handed.

FEEDS INTO:
    app.py (the Streamlit dashboard), via the history log this file
    writes to disk.

WHY THIS SPLIT MATTERS (design principle, not just a code detail):
    The LLM is deliberately kept as a NARRATOR, not a DETECTOR. All the
    actual statistical work (where did the regime change, what state
    are we in) happens in detect_regime.py using validated, testable
    methods. The LLM's only input is that finished analysis -- it
    cannot invent a regime that wasn't actually detected, because it's
    never shown raw data to reinterpret.
=====================================================================
"""
import os
import json
from datetime import datetime
from groq import Groq


SYSTEM_PROMPT = """You are writing a short, friendly explainer for a public dashboard \
called Regime Watch, aimed at someone with NO background in statistics, finance, or \
data science. Assume the reader is smart but has never heard of a "standard deviation," \
a "Hidden Markov Model," or a "regime label," and never will.

You are given structured statistical output from a regime-detection pipeline. Your job \
is to translate it, not report it.

Hard rules:
- NEVER use the words: regime label, state, HMM, Hidden Markov, standard deviation, \
mean, variance, volatility (as a technical term), or any numeric state index like \
"regime 0" or "regime 1".
- NEVER quote a raw statistical value like a mean or std directly (e.g. do not write \
"a mean of 4.32776"). Instead, describe what it MEANS in everyday terms (e.g. "interest \
rates have been unusually high for an extended stretch").
- Do not use hedging clichés ("it's important to note", "in today's ever-changing landscape").
- No em dashes.
- Write 100-150 words.
- Structure: (1) one sentence on what's happening right now, in plain terms, (2) one or \
two sentences on how that compares to the past, using relatable language ("this is one \
of the calmer stretches in recent years" rather than exact numbers), (3) one sentence of \
real-world context if relevant (e.g. connecting a rate rise to inflation-fighting, if \
that's a reasonable general inference), avoiding invented specifics not given to you.
- You MAY use the actual current value (e.g. "currently at 3.75%") since that's a real,
concrete, easy-to-understand number -- just avoid statistical jargon around it.
- Do not predict a specific future value. You can note how long similar past stretches \
have lasted, framed as historical context, not a promise.
- Write like you're explaining this to a smart friend over coffee, not writing a report.
"""


def generate_brief(analysis: dict) -> str:
    """
    INPUT:
        analysis -- the dictionary returned by detect_regime.analyze_series().
                    Contains only finished facts (dates, means, std devs,
                    labels) -- no raw time series data.

    PROCESS:
        1. Create a Groq API client using an API key read from the
           environment (never hard-coded, so the key is never
           accidentally committed to the repo).
        2. Build a user prompt that lists out every relevant fact from
           the analysis dictionary in plain key: value form.
        3. Send both the SYSTEM_PROMPT (the fixed rules the model
           should always follow) and the user prompt (this specific
           series' facts) to the chat completions endpoint.
        4. Extract just the text of the model's reply.

    OUTPUT:
        A string: the plain-English brief, 120-180 words per the
        system prompt's instruction.

    REASON THIS APPROACH:
        The prompt explicitly forbids inventing a future prediction --
        this mirrors the honest boundary we validated earlier: regime
        detection tells you about the PRESENT state and PAST comparison,
        not a reliable future price. Keeping that same honesty
        constraint in the LLM's instructions prevents the narration
        layer from overselling what the detection layer never actually
        promised.
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    user_prompt = f"""Series: {analysis['series_name']}
Detection mode: {analysis['mode']}
Latest value: {analysis['latest_value']} (as of {analysis['latest_date']})
Current regime label: {analysis['current_regime_label']}
Current regime has held since: {analysis['regime_since']} ({analysis['days_in_current_regime']} days)
Current regime mean: {analysis['regime_mean']:.5f}
Current regime volatility (std): {analysis['regime_std']:.5f}
All historical regime states (for comparison): {json.dumps(analysis['all_states'])}
Detected structural change-point dates: {analysis['changepoint_dates']}

Write the brief now."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=400,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def save_brief_to_history(analysis: dict, brief_text: str, history_path: str = "data/history.json") -> None:
    """
    INPUT:
        analysis -- the same analysis dictionary passed to generate_brief.
        brief_text -- the string generate_brief() returned.
        history_path -- where the running log file lives on disk.

    PROCESS:
        1. Try to read the existing history file and parse it as JSON.
           If the file doesn't exist yet, or is somehow corrupted,
           start a fresh empty list rather than crashing.
        2. Build one new entry: a timestamp, the series name, the full
           analysis dict, and the brief text.
        3. Append it to the list and write the whole list back to disk.

    OUTPUT:
        None returned -- the side effect is an updated history.json
        file on disk, with one more entry than before.

    REASON THIS APPROACH:
        This file is what makes the project "ongoing" rather than a
        one-off script: every time the pipeline runs (weekly, via
        GitHub Actions), a new dated entry gets appended, building up a
        real history over time. This is also the foundation for the
        planned accuracy tracker: because every past call is logged
        with its own stats, a future script can look back and check
        whether a regime's implied volatility actually held up over the
        following weeks.
    """
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    entry = {
        "generated_at": datetime.utcnow().isoformat(),
        "series_name": analysis["series_name"],
        "analysis": analysis,
        "brief": brief_text,
    }
    history.append(entry)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)


if __name__ == "__main__":
    # Manual smoke-test entry point: run this file directly to check the
    # Groq connection works, using a hand-written example analysis dict
    # rather than requiring the full fetch+detect pipeline to run first.
    example_analysis = {
        "series_name": "bank_rate",
        "mode": "level",
        "latest_value": 4.75,
        "latest_date": "2026-07-01",
        "current_regime_label": 1,
        "regime_since": "2021-12-16",
        "days_in_current_regime": 1200,
        "regime_mean": 3.98,
        "regime_std": 1.28,
        "all_states": {"0": {"mean": 0.40, "std": 0.22, "n_points": 1656}},
        "changepoint_dates": ["2016-08-02", "2020-03-13", "2021-12-16"],
    }
    brief = generate_brief(example_analysis)
    print(brief)