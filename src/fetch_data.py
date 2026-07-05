"""
=====================================================================
 FILE: fetch_data.py
=====================================================================
ROLE IN THE PIPELINE:
    This is the FIRST stage of Regime Watch. It has exactly one job:
    pull raw data from three external sources and save it as clean,
    consistent CSV files on disk.

DEPENDS ON:
    Nothing else in this project. It only talks to external services
    (Bank of England, ONS, Yahoo Finance).

FEEDS INTO:
    detect_regime.py -- which reads the CSVs this file produces and
    never talks to any external API itself. This separation matters:
    if BoE's website changes its format tomorrow, only THIS file needs
    fixing, not the analysis logic.

DATA SOURCES AND WHY EACH IS FETCHED DIFFERENTLY:
    1. Bank of England Bank Rate -- no real API, just a CSV-download
       URL trick (validated against BoE's actual site during development).
    2. ONS CPI inflation -- a proper JSON REST API.
    3. GBP/USD -- via the yfinance package, which wraps Yahoo Finance.

    Each source needed its own handling because each returns data in a
    genuinely different shape (HTML-or-CSV text, JSON, or a library's
    own DataFrame) -- there's no way to write one generic fetcher for
    all three honestly.
=====================================================================
"""
import io
import csv
import os
import requests
import pandas as pd
import yfinance as yf


def fetch_boe_bank_rate(date_from: str = "01/Jan/2010", date_to: str = "01/Jan/2030") -> pd.DataFrame:
    """
    INPUT:
        date_from, date_to -- strings in 'DD/Mon/YYYY' format, the date
        range to request from BoE.

    PROCESS:
        1. Build the query parameters BoE's database endpoint expects.
           SeriesCodes='IUDBEDR' is specifically the Bank Rate series
           (found via BoE's own database search tool).
        2. Send a GET request with a browser-like User-Agent header,
           because BoE's server blocks requests that look automated.
        3. Defensively check whether the response is actually CSV data
           or an HTML error page -- BoE returns error pages with a
           normal HTTP 200 status, so a status check alone wouldn't
           catch a failed request.
        4. Parse the CSV text into a DataFrame, rename columns to
           something readable, and convert the date text (e.g.
           "02 Aug 2016") into real datetime objects.

    OUTPUT:
        A DataFrame with columns ['date', 'bank_rate'], sorted
        chronologically, with any unparseable rows dropped.

    REASON THIS APPROACH:
        BoE has no formal REST API. This CSV-download endpoint is the
        only practical way to get this data without manual downloads.
        The HTML-sniffing check exists because we discovered during
        testing that failed requests don't look like failures at the
        HTTP level -- they look like successful requests that happen
        to return a webpage instead of data.
    """
    url = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
    params = {
        "csv.x": "yes",
        "Datefrom": date_from,
        "Dateto": date_to,
        "SeriesCodes": "IUDBEDR",
        "CSVF": "TN",
        "UsingCodes": "Y",
        "VPD": "Y",
        "VFD": "N",
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=15)

    if "<html" in resp.text.lower()[:200]:
        raise ValueError(
            "BoE returned an HTML page instead of CSV. "
            "Check series code / date format, or the endpoint may be temporarily down."
        )

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "bank_rate"]
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_ons_cpi() -> pd.DataFrame:
    """
    INPUT:
        None (the series ID is fixed: D7OE under dataset MM23, which is
        the CPI monthly rate -- found via the ONS time series explorer).

    PROCESS:
        1. Download the current MM23 dataset CSV from ONS using a
           browser-like User-Agent header.
        2. Read the wide export, locate the D7OE series column, and
           keep only rows where that series has an observation.
        3. Convert the year labels into real datetimes and rename the
           value column to "cpi_rate" for downstream consistency.

    OUTPUT:
        A DataFrame with columns ['date', 'cpi_rate'], sorted
        chronologically.

    REASON THIS APPROACH:
        The old ONS timeseries endpoint now returns 404. The dataset CSV
        download exposed on the live ONS page is stable and contains the
        current series definitions, so reading that export avoids the
        stale explorer route while keeping the downstream schema the
        same.
    """
    url = "https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindices/current/mm23.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    rows = list(csv.reader(io.StringIO(resp.text)))
    if len(rows) < 8:
        raise ValueError("ONS CPI dataset export is missing expected header rows.")

    try:
        series_col = rows[1].index("D7OE")
    except ValueError as exc:
        raise ValueError("ONS CPI dataset export no longer contains series D7OE.") from exc

    records = []
    for row in rows[7:]:
        if len(row) <= series_col:
            continue
        date_label = row[0].strip()
        value_text = row[series_col].strip()
        if not date_label or not value_text:
            continue
        records.append({"date": date_label, "cpi_rate": value_text})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("ONS CPI dataset export returned no annual rate observations.")

    def parse_ons_label(label: str) -> pd.Timestamp:
        label = label.strip()
        if len(label) == 4 and label.isdigit():
            return pd.to_datetime(label, format="%Y")
        return pd.to_datetime(label, format="%Y %b")

    df["date"] = df["date"].map(parse_ons_label)
    df["cpi_rate"] = df["cpi_rate"].astype(float)
    return df[["date", "cpi_rate"]].sort_values("date").reset_index(drop=True)


def fetch_gbpusd(period: str = "5y") -> pd.DataFrame:
    """
    INPUT:
        period -- how far back to fetch, in yfinance's own shorthand
        (e.g. "5y" = 5 years). Kept as a string because that's the
        format yfinance itself expects.

    PROCESS:
        1. Call yfinance's download function for the GBPUSD=X ticker.
        2. Keep only the 'Close' column -- yfinance also returns
           Open/High/Low/Volume, which this project doesn't use.
        3. Move the date out of the DataFrame's index and into a
           normal column, matching the shape of the other two fetchers
           (so all three CSVs have a consistent ['date', <value>] shape).

    OUTPUT:
        A DataFrame with columns ['date', 'gbpusd'].

    REASON THIS APPROACH:
        yfinance already handles the messy details of talking to Yahoo
        Finance, so this is intentionally the simplest of the three
        functions. Keeping only 'Close' and renaming columns is purely
        about keeping every fetched series in the SAME shape, so
        detect_regime.py can treat any of them identically downstream.
    """
    raw = yf.download("GBPUSD=X", period=period, interval="1d", progress=False)
    if raw.empty:
        raise ValueError("yfinance returned no data for GBPUSD=X — check ticker or network access.")

    df = raw[["Close"]].reset_index()
    df.columns = ["date", "gbpusd"]
    return df


def fetch_all(save_dir: str = "data/raw") -> None:
    """
    INPUT:
        save_dir -- folder path where the three CSVs will be written.

    PROCESS:
        Calls each of the three fetch functions in turn, saves each
        result to its own CSV, and prints a short sanity-check summary
        (row count + date range) after each one.

    OUTPUT:
        None returned -- this function's job is the SIDE EFFECT of
        writing three files to disk: bank_rate.csv, cpi.csv, gbpusd.csv.

    REASON THIS APPROACH:
        This is the single entry point the rest of the pipeline calls
        (from run_pipeline.py, and from the GitHub Actions workflow).
        The printed summary after each fetch exists because a silent
        failure here (e.g. an empty or truncated file) would be very
        hard to notice until much later, downstream in the analysis --
        printing row counts and date ranges immediately makes an
        obviously broken fetch visible right away.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("Fetching Bank Rate...")
    bank_rate = fetch_boe_bank_rate()
    bank_rate.to_csv(f"{save_dir}/bank_rate.csv", index=False)
    print(f"  {len(bank_rate)} rows, {bank_rate['date'].min().date()} to {bank_rate['date'].max().date()}")

    print("Fetching CPI...")
    cpi = fetch_ons_cpi()
    cpi.to_csv(f"{save_dir}/cpi.csv", index=False)
    print(f"  {len(cpi)} rows, {cpi['date'].min().date()} to {cpi['date'].max().date()}")

    print("Fetching GBP/USD...")
    fx = fetch_gbpusd()
    fx.to_csv(f"{save_dir}/gbpusd.csv", index=False)
    print(f"  {len(fx)} rows, {fx['date'].min().date()} to {fx['date'].max().date()}")

    print("Done.")


if __name__ == "__main__":
    fetch_all()