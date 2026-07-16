#!/usr/bin/env python3
"""Small, retrying Yahoo price exporter for personal strategy research.

One-symbol requests keep failures auditable and prevent one renamed ticker from
invalidating a whole batch. Artifacts are short-lived and are never committed.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yfinance as yf

MISSING_FUTU = [
    "PEP", "PFE", "PG", "PLTR", "PM", "RTX", "SBUX", "SCHW",
    "SO", "SPG", "T", "TMO", "TMUS", "TSLA", "UBER", "UNH",
    "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]
PROBE = ["SPY", "AAPL", "TSLA", "XOM"]
ALIASES = {"BRK.B": ["BRK-B"], "BNY": ["BNY", "BK"], "HONA": ["HONA", "HON"]}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=["probe", "missing_2019_plus"], default="probe")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None, help="Exclusive end date, as used by yfinance")
    p.add_argument("--out-dir", default="yahoo-export")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--pause", type=float, default=2.0)
    return p.parse_args()

def clean_history(frame, canonical, yahoo_symbol):
    if frame is None or frame.empty:
        return pd.DataFrame()
    df = frame.copy().reset_index()
    date_col = "Date" if "Date" in df.columns else "Datetime"
    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    out = pd.DataFrame({"date": dates})
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]:
        if col not in df.columns:
            df[col] = 0.0 if col in {"Dividends", "Stock Splits"} else pd.NA
    raw_close = pd.to_numeric(df["Close"], errors="coerce")
    adj_close = pd.to_numeric(df["Adj Close"], errors="coerce").fillna(raw_close)
    factor = (adj_close / raw_close.replace(0, pd.NA)).fillna(1.0)
    out["symbol"] = canonical
    out["source_symbol"] = yahoo_symbol
    out["open"] = pd.to_numeric(df["Open"], errors="coerce") * factor
    out["high"] = pd.to_numeric(df["High"], errors="coerce") * factor
    out["low"] = pd.to_numeric(df["Low"], errors="coerce") * factor
    out["close"] = adj_close
    out["raw_close"] = raw_close
    out["volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    out["turnover"] = raw_close * out["volume"]
    out["dividends"] = pd.to_numeric(df["Dividends"], errors="coerce").fillna(0.0)
    out["stock_splits"] = pd.to_numeric(df["Stock Splits"], errors="coerce").fillna(0.0)
    return out.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")

def download_one(yahoo_symbol, start, end, retries):
    errors = []
    waits = [5, 15, 45]
    for attempt in range(retries):
        try:
            frame = yf.Ticker(yahoo_symbol).history(
                start=start, end=end, interval="1d", auto_adjust=False,
                actions=True, repair=True, timeout=30,
            )
            if frame is not None and not frame.empty:
                return frame, errors
            errors.append(f"attempt {attempt + 1}: empty response")
        except Exception as exc:
            errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
        if attempt + 1 < retries:
            time.sleep(waits[min(attempt, len(waits) - 1)])
    return pd.DataFrame(), errors

def main():
    args = parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    symbols = PROBE if args.scope == "probe" else ["SPY", *MISSING_FUTU]
    frames, results = [], {}
    for index, canonical in enumerate(symbols):
        pieces, all_errors = [], {}
        for yahoo_symbol in ALIASES.get(canonical, [canonical]):
            raw, errors = download_one(yahoo_symbol, args.start, args.end, args.retries)
            all_errors[yahoo_symbol] = errors
            cleaned = clean_history(raw, canonical, yahoo_symbol)
            if not cleaned.empty:
                pieces.append(cleaned)
        if pieces:
            combined = pd.concat(list(reversed(pieces)), ignore_index=True)
            combined = combined.drop_duplicates("date", keep="last").sort_values("date")
            frames.append(combined)
            results[canonical] = {
                "status": "ok", "rows": int(len(combined)),
                "first_date": combined["date"].min().date().isoformat(),
                "last_date": combined["date"].max().date().isoformat(),
                "source_symbols": sorted(combined["source_symbol"].unique().tolist()),
                "errors": all_errors,
            }
        else:
            results[canonical] = {"status": "failed", "rows": 0, "errors": all_errors}
        print(json.dumps({canonical: results[canonical]}, ensure_ascii=False), flush=True)
        if index + 1 < len(symbols):
            time.sleep(args.pause)

    exported = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not exported.empty:
        exported = exported.sort_values(["date", "symbol"])
        exported["date"] = exported["date"].dt.date.astype(str)
        exported.to_csv(output / "prices_yahoo.csv", index=False)
        if "SPY" in set(exported["symbol"]):
            spy_dates = set(exported.loc[exported["symbol"] == "SPY", "date"])
            for symbol, group in exported.groupby("symbol"):
                first = group["date"].min()
                expected = {d for d in spy_dates if d >= first}
                present = set(group["date"])
                results[symbol]["missing_vs_spy_after_first"] = sorted(expected - present)
                results[symbol]["missing_count"] = len(expected - present)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "scope": args.scope, "start": args.start, "end_exclusive": args.end,
        "symbols": symbols, "results": results,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ok = sum(item["status"] == "ok" for item in results.values())
    print(f"completed: {ok}/{len(results)} symbols")
    return 0 if ok else 2

if __name__ == "__main__":
    sys.exit(main())
