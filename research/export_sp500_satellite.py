"""Export one resumable shard of Yahoo daily history for S&P 500 satellites.

Designed for GitHub Actions, where a fresh runner has proven more reliable than
the local Yahoo session.  Every failed ticker remains in the manifest; failures
are never silently dropped from the research universe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


OUTPUT_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "last_close",
    "raw_close",
    "dividends",
    "stock_splits",
    "source_symbol",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_symbol_map(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    required = {"symbol", "yahoo_symbol"}
    if not required.issubset(frame.columns):
        raise ValueError(f"symbol map needs {sorted(required)}")
    frame = frame[list(required)].copy()
    frame["symbol"] = frame["symbol"].str.strip().str.upper()
    frame["yahoo_symbol"] = frame["yahoo_symbol"].str.strip().str.upper()
    frame = frame.replace("", np.nan).dropna().drop_duplicates()
    collisions = frame.groupby("yahoo_symbol")["symbol"].nunique()
    collisions = collisions[collisions > 1]
    if len(collisions):
        raise ValueError(f"Yahoo symbol collisions: {collisions.to_dict()}")
    return frame.sort_values(["yahoo_symbol", "symbol"]).reset_index(drop=True)


def choose_shard(frame: pd.DataFrame, index: int, count: int) -> pd.DataFrame:
    if count < 1 or index < 0 or index >= count:
        raise ValueError("shard index must satisfy 0 <= index < count")
    return frame.iloc[[i % count == index for i in range(len(frame))]].reset_index(drop=True)


def normalize_history(raw: pd.DataFrame, symbol: str, yahoo_symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data = raw.copy().reset_index()
    date_col = next((c for c in data.columns if str(c).lower() in {"date", "datetime"}), None)
    if date_col is None:
        raise ValueError("Yahoo history has no Date/Datetime column")

    date_values = pd.to_datetime(data[date_col], errors="coerce", utc=True)
    data["date"] = date_values.dt.tz_convert(None).dt.normalize()
    rename = {
        "Open": "raw_open",
        "High": "raw_high",
        "Low": "raw_low",
        "Close": "raw_close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "stock_splits",
    }
    data = data.rename(columns=rename)
    for column in ("raw_open", "raw_high", "raw_low", "raw_close", "volume"):
        if column not in data:
            raise ValueError(f"Yahoo history missing {column}")
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if "adj_close" not in data:
        data["adj_close"] = data["raw_close"]
    data["adj_close"] = pd.to_numeric(data["adj_close"], errors="coerce")
    for column in ("dividends", "stock_splits"):
        if column not in data:
            data[column] = 0.0
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)

    factor = data["adj_close"] / data["raw_close"].replace(0, np.nan)
    factor = factor.where(np.isfinite(factor) & factor.gt(0), 1.0)
    output = pd.DataFrame(
        {
            "date": data["date"],
            "symbol": symbol,
            "open": data["raw_open"] * factor,
            "high": data["raw_high"] * factor,
            "low": data["raw_low"] * factor,
            "close": data["adj_close"],
            "volume": data["volume"].fillna(0).round().astype("Int64"),
            "turnover": data["raw_close"] * data["volume"],
            "last_close": data["adj_close"].shift(1),
            "raw_close": data["raw_close"],
            "dividends": data["dividends"],
            "stock_splits": data["stock_splits"],
            "source_symbol": yahoo_symbol,
        }
    )
    output = output.dropna(subset=["date", "open", "high", "low", "close"])
    output = output.sort_values("date").drop_duplicates("date", keep="last")
    return output[OUTPUT_COLUMNS].reset_index(drop=True)


def fetch_one(
    symbol: str,
    yahoo_symbol: str,
    start: str,
    end: str,
    retries: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    for attempt in range(1, retries + 1):
        try:
            raw = yf.Ticker(yahoo_symbol).history(
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                actions=True,
                repair=True,
                raise_errors=True,
            )
            normalized = normalize_history(raw, symbol, yahoo_symbol)
            if normalized.empty:
                raise ValueError("empty normalized history")
            attempts.append({"attempt": attempt, "status": "ok", "rows": len(normalized)})
            return normalized, attempts
        except Exception as exc:  # manifest retains the exact failure
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
            )
            if attempt < retries:
                time.sleep(min(60, 5 * (3 ** (attempt - 1))))
    return pd.DataFrame(columns=OUTPUT_COLUMNS), attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols-file", required=True)
    parser.add_argument("--start", default="2018-07-01")
    parser.add_argument("--end", default="2026-07-18", help="Yahoo-exclusive end date")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.35)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    source_path = Path(args.symbols_file)
    out_dir = Path(args.out_dir)
    per_symbol = out_dir / "symbols"
    per_symbol.mkdir(parents=True, exist_ok=True)

    all_symbols = load_symbol_map(source_path)
    shard = choose_shard(all_symbols, args.shard_index, args.shard_count)
    print(
        f"shard {args.shard_index}/{args.shard_count}: {len(shard)} of "
        f"{len(all_symbols)} symbols",
        flush=True,
    )

    records: list[dict[str, object]] = []
    chunks: list[pd.DataFrame] = []
    for ordinal, row in enumerate(shard.itertuples(index=False), start=1):
        frame, attempts = fetch_one(
            row.symbol, row.yahoo_symbol, args.start, args.end, args.retries
        )
        status = "ok" if len(frame) else "failed"
        if len(frame):
            target = per_symbol / f"{row.symbol.replace('.', '_')}.csv"
            frame.to_csv(target, index=False, date_format="%Y-%m-%d")
            chunks.append(frame)
        records.append(
            {
                "symbol": row.symbol,
                "yahoo_symbol": row.yahoo_symbol,
                "status": status,
                "rows": int(len(frame)),
                "first_date": frame["date"].min().date().isoformat() if len(frame) else None,
                "last_date": frame["date"].max().date().isoformat() if len(frame) else None,
                "attempts": attempts,
            }
        )
        print(
            f"[{ordinal:03d}/{len(shard):03d}] {row.symbol:<8} {status:<6} "
            f"rows={len(frame)}",
            flush=True,
        )
        if ordinal < len(shard):
            time.sleep(max(0.0, args.pause))

    prices = (
        pd.concat(chunks, ignore_index=True)
        .sort_values(["date", "symbol"])
        .drop_duplicates(["date", "symbol"], keep="last")
        if chunks
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    prices.to_csv(out_dir / "prices.csv", index=False, date_format="%Y-%m-%d")
    failed = [item["symbol"] for item in records if item["status"] != "ok"]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "source": "Yahoo Finance via lonercai0808/yfinance GitHub runner",
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "start": args.start,
        "end_exclusive": args.end,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "symbol_map_sha256": sha256(source_path),
        "requested_symbols": int(len(shard)),
        "successful_symbols": int(len(shard) - len(failed)),
        "failed_symbols": failed,
        "rows": int(len(prices)),
        "duplicate_date_symbol_rows": int(prices.duplicated(["date", "symbol"]).sum()),
        "records": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2))
    if len(failed) == len(shard) and len(shard):
        raise SystemExit("Every symbol failed; preserving manifest and artifact")


if __name__ == "__main__":
    main()
