# vintages.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from DFM_helpers import load_dfm_ready, load_release_lags_csv, make_vintage

logger = logging.getLogger("VINTAGES")

# -----------------------------
# Config (edit paths)
# -----------------------------
DATA_PATH = Path("/Users/babsvanderlugt/Downloads/seminar vs code/Seminar-GDP-nowcasting-11/data/data_transformations_DFM_ready_state_space.csv")
LAGS_PATH = Path("/Users/babsvanderlugt/Downloads/seminar vs code/Seminar-GDP-nowcasting-11/release_lags_clean.csv")

OUT_DIR = Path("vintages_store")      # folder where vintages will be written
OUT_FORMAT = "parquet"                # "parquet" (recommended) or "csv"

ASOF_START = "2005-01-31"
ASOF_END   = "2025-05-31"
ASOF_FREQ  = "ME"  # Month-End


def month_end_grid(df_index: pd.DatetimeIndex, start=None, end=None) -> List[pd.Timestamp]:
    start = df_index.min().to_period("M").to_timestamp(how="end") if start is None else pd.to_datetime(start)
    end   = df_index.max().to_period("M").to_timestamp(how="end") if end is None else pd.to_datetime(end)
    grid = pd.date_range(start=start, end=end, freq=ASOF_FREQ)
    return [pd.Timestamp(x) for x in grid]


def vintage_filename(as_of: pd.Timestamp, ext: str) -> str:
    return f"vintage_{as_of.strftime('%Y-%m-%d')}.{ext}"


def save_vintage(v: pd.DataFrame, as_of: pd.Timestamp, out_dir: Path, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower()
    if fmt == "parquet":
        out_path = out_dir / vintage_filename(as_of, "parquet")
        try:
            v.to_parquet(out_path, index=True)  # requires pyarrow or fastparquet
            return out_path
        except ImportError:
            # fallback if parquet engine not installed
            fmt = "csv"

    if fmt == "csv":
        out_path = out_dir / vintage_filename(as_of, "csv")
        v.to_csv(out_path, index=True)
        return out_path

    raise ValueError("OUT_FORMAT must be 'parquet' or 'csv'.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    logger.info("Loading base panel + release lags...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH}")
    if not LAGS_PATH.exists():
        raise FileNotFoundError(f"LAGS_PATH not found: {LAGS_PATH}")

    df = load_dfm_ready(DATA_PATH, freq="MS")
    lags = load_release_lags_csv(LAGS_PATH)

    asofs = month_end_grid(df.index, start=ASOF_START, end=ASOF_END)
    logger.info(f"As-of dates: n={len(asofs)} | {asofs[0].date()} .. {asofs[-1].date()}")
    logger.info(f"Saving vintages to: {OUT_DIR.resolve()} ({OUT_FORMAT})")

    manifest_rows = []

    for i, as_of in enumerate(asofs, start=1):
        logger.info(f"[{i}/{len(asofs)}] Building vintage for as_of={as_of.date()}")

        v = make_vintage(df, as_of=as_of, use_real_lags=True, lags=lags, verbose=False)

        out_path = save_vintage(v, as_of=as_of, out_dir=OUT_DIR, fmt=OUT_FORMAT)

        manifest_rows.append(
            {
                "as_of": as_of.strftime("%Y-%m-%d"),
                "file": str(out_path),
                "n_rows": int(v.shape[0]),
                "n_cols": int(v.shape[1]),
                "na_share": float(v.isna().mean().mean()),
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values("as_of")
    manifest_path = OUT_DIR / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    logger.info(f"Done. Manifest saved -> {manifest_path}")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
