from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


DATA_ROOT = Path(r"C:\Users\noran\Documents\Documents\ERASMUS\MASTER\SEMINAR\Erasmus_GetData 2\Erasmus_GetData 2\data")
CODE_ROOT = Path(r"C:\Users\noran\Documents\Documents\ERASMUS\MASTER\SEMINAR\Erasmus_GetData 2\Erasmus_GetData 2\data\a01_get_data\get_data_code")

TARGET_COLS = [
    "Industrial_confidence",
    "BusinessLead",
    "Construction__412"
]

CASE_SENSITIVE = True


ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
SEPS_TO_TRY = [",", ";", "\t", "|"]


@dataclass(frozen=True)
class PyHit:
    py_file: Path
    line_no: int
    line: str


def iter_csv_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() == ".csv":
                out.append(p)
    return out


def iter_py_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(Path(dirpath) / fn)
    return out


def normalize(s: str) -> str:
    s = str(s).strip()
    # jouw lijst heeft '"Crude oil, Brent"' met quotes, csv header vaak zonder
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1]
    return s


def build_target_set(targets: List[str]) -> Set[str]:
    if CASE_SENSITIVE:
        return {normalize(t) for t in targets}
    return {normalize(t).lower() for t in targets}


def find_matching_cols(cols: List[str], target_set: Set[str]) -> List[str]:
    found: List[str] = []
    for c in cols:
        cc = normalize(c)
        key = cc if CASE_SENSITIVE else cc.lower()
        if key in target_set:
            found.append(cc)
    return found


def try_read_header_with_pandas(csv_path: Path) -> Optional[List[str]]:
    """
    Try multiple encodings and separators to read header only.
    Returns list of columns if success else None.
    """
    for enc in ENCODINGS_TO_TRY:
        for sep in SEPS_TO_TRY:
            try:
                df = pd.read_csv(
                    csv_path,
                    nrows=0,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip",
                )
                cols = [str(c).strip() for c in df.columns.tolist()]
                # Heuristic: if only 1 "column" and it contains sep or looks weird, probably wrong sep
                if len(cols) == 1 and (sep in cols[0]):
                    continue
                return cols
            except Exception:
                continue
    return None


def brute_header_fallback(csv_path: Path) -> Optional[List[str]]:
    """
    Last resort: read first few lines as text and try to interpret first non-empty line as header,
    splitting by common separators.
    """
    for enc in ENCODINGS_TO_TRY:
        try:
            text = csv_path.read_text(encoding=enc, errors="replace")
            lines = [ln.strip("\n\r") for ln in text.splitlines()[:20]]
            # find first non-empty line
            header_line = None
            for ln in lines:
                if ln.strip():
                    header_line = ln
                    break
            if not header_line:
                return None

            # try split by candidate separators and pick the split that gives most fields
            best = None
            best_n = 0
            for sep in SEPS_TO_TRY:
                parts = [p.strip() for p in header_line.split(sep)]
                if len(parts) > best_n:
                    best_n = len(parts)
                    best = parts

            if best and best_n >= 2:
                return best
        except Exception:
            continue
    return None


def read_csv_columns_robust(csv_path: Path) -> Optional[List[str]]:
    cols = try_read_header_with_pandas(csv_path)
    if cols is not None:
        return cols
    return brute_header_fallback(csv_path)


def read_py_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def find_stem_in_python(stem: str, py_files: List[Path]) -> List[PyHit]:
    hits: List[PyHit] = []
    for py in py_files:
        for i, line in enumerate(read_py_lines(py), start=1):
            if stem in line:
                hits.append(PyHit(py_file=py, line_no=i, line=line.strip()))
    return hits


def main():
    if not DATA_ROOT.exists():
        raise SystemExit(f"DATA_ROOT bestaat niet: {DATA_ROOT}")
    if not CODE_ROOT.exists():
        raise SystemExit(f"CODE_ROOT bestaat niet: {CODE_ROOT}")

    target_set = build_target_set(TARGET_COLS)

    csv_files = iter_csv_files(DATA_ROOT)
    py_files = iter_py_files(CODE_ROOT)

    print(f"CSV files gescand: {len(csv_files)}")
    print(f"Python files gescand: {len(py_files)}")

    csv_matches: Dict[Path, List[str]] = {}
    unreadable: List[Path] = []

    for csv in csv_files:
        cols = read_csv_columns_robust(csv)
        if cols is None:
            unreadable.append(csv)
            continue
        found = find_matching_cols(cols, target_set)
        if found:
            csv_matches[csv] = found

    print("\n" + "=" * 100)
    print("STAP 1: CSV files met minstens 1 van jouw kolomnamen")
    print("=" * 100)

    if not csv_matches:
        print("Nog steeds niks gevonden.")
        print("Dan is 1 van deze 3 dingen waar:")
        print("  1) De kolomnamen staan niet in CSV headers (maar in data-rijen of in aparte mapping)")
        print("  2) Je kolomnamen zitten niet in deze DATA_ROOT (misschien andere folder)")
        print("  3) Kolomnamen wijken af (spaties/underscores/typos)")
    else:
        for csv, cols in sorted(csv_matches.items(), key=lambda x: str(x[0])):
            print(f"\n{csv}")
            print(f"  gevonden kolommen ({len(cols)}): {cols}")

    if unreadable:
        print("\n" + "-" * 100)
        print(f"LET OP: {len(unreadable)} CSV files kon ik niet lezen.")
        print("Eerste 10:")
        for f in unreadable[:10]:
            print(f"  - {f}")

    if not csv_matches:
        return

    print("\n" + "=" * 100)
    print("STAP 2: Zoek CSV bestandsnaam ZONDER .csv in Python code")
    print("=" * 100)

    any_hits = False
    for csv in sorted(csv_matches.keys(), key=str):
        stem = csv.stem
        hits = find_stem_in_python(stem, py_files)

        print("\n" + "-" * 100)
        print(f"CSV: {csv.name}  -> zoekterm: '{stem}'")
        if not hits:
            print("  Geen hits in python code.")
            continue

        any_hits = True
        for h in hits:
            print(f"  - {h.py_file}  L{h.line_no}: {h.line}")

    if not any_hits:
        print("\nGeen enkele CSV-stem kwam letterlijk voor in de python code.")
        print("Dan worden filenames vrijwel zeker dynamisch gebouwd.")
        print("In dat geval moet je zoeken op to_csv( / read_csv( / output_dir variabelen.")


if __name__ == "__main__":
    main()
