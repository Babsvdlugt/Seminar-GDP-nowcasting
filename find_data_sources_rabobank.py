from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


DATA_ROOT = Path(r"C:\Users\noran\Documents\Documents\ERASMUS\MASTER\SEMINAR\Erasmus_GetData 2\Erasmus_GetData 2\data")
CODE_ROOT = Path(r"C:\Users\noran\Documents\Documents\ERASMUS\MASTER\SEMINAR\Erasmus_GetData 2\Erasmus_GetData 2\data\a01_get_data\get_data_code")

TARGET_COLS = [
    "Util", 
    "House", 
    "Lead", 
    "Confidence",
    "Exchange", 
    "Wissel",
    "Monitor",
    "CPI",
]

CASE_SENSITIVE = True  #hoofdlettergevoeligheid

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
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1]
    return s


def keyify(s: str) -> str:
    s = normalize(s)
    return s if CASE_SENSITIVE else s.lower()


def build_target_set(targets: List[str]) -> Set[str]:
    return {keyify(t) for t in targets}


def try_read_header_with_pandas(csv_path: Path) -> Optional[Tuple[List[str], str, str]]:
    """
    Returns (cols, encoding, sep) if success else None
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
                if len(cols) == 1 and (sep in cols[0]):
                    continue
                return cols, enc, sep
            except Exception:
                continue
    return None


def brute_header_fallback(csv_path: Path) -> Optional[Tuple[List[str], str, str]]:
    """
    Last resort: read as text, find first non-empty line, split by best sep.
    Returns (cols, encoding, sep) if success else None
    """
    for enc in ENCODINGS_TO_TRY:
        try:
            text = csv_path.read_text(encoding=enc, errors="replace")
            lines = [ln.strip("\n\r") for ln in text.splitlines()[:40]]
            header_line = None
            for ln in lines:
                if ln.strip():
                    header_line = ln
                    break
            if not header_line:
                return None

            best_parts = None
            best_sep = None
            best_n = 0
            for sep in SEPS_TO_TRY:
                parts = [p.strip() for p in header_line.split(sep)]
                if len(parts) > best_n:
                    best_n = len(parts)
                    best_parts = parts
                    best_sep = sep

            if best_parts and best_n >= 2 and best_sep is not None:
                return best_parts, enc, best_sep
        except Exception:
            continue
    return None


def read_csv_columns_robust(csv_path: Path) -> Optional[Tuple[List[str], str, str]]:
    got = try_read_header_with_pandas(csv_path)
    if got is not None:
        return got
    return brute_header_fallback(csv_path)


def build_header_line(cols: List[str], sep: str) -> str:
    # puur voor printbaarheid, niet voor parsing
    return sep.join(cols)


def read_py_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def find_in_python(needle: str, py_files: List[Path]) -> List[PyHit]:
    hits: List[PyHit] = []
    if not needle:
        return hits

    for py in py_files:
        for i, line in enumerate(read_py_lines(py), start=1):
            hay = line if CASE_SENSITIVE else line.lower()
            ndl = needle if CASE_SENSITIVE else needle.lower()
            if ndl in hay:
                hits.append(PyHit(py_file=py, line_no=i, line=line.strip()))
    return hits


def short_loc(p: Path, root: Path) -> str:
    """
    Output like: filename.ext | mapnaam
    mapnaam = de folder direct onder root (of '.' als hij direct onder root zit)
    """
    try:
        rel = p.relative_to(root)
        parts = rel.parts
        folder = parts[0] if len(parts) >= 2 else "."
    except Exception:
        folder = p.parent.name or "."
    return f"{p.name} | {folder}"


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

    # Index: target_key -> list of (csv_path, header_line)
    hits_by_target: Dict[str, List[Tuple[Path, str]]] = {keyify(t): [] for t in TARGET_COLS}
    unreadable: List[Path] = []

    for csv in csv_files:
        got = read_csv_columns_robust(csv)
        if got is None:
            unreadable.append(csv)
            continue

        cols, _, sep = got
        cols_norm = [normalize(c) for c in cols]

        # maak 1 printbare header-regel
        header_line = build_header_line(cols_norm, sep)

        # check per target
        for t in target_set:
            found_this_t = False
            for c in cols_norm:
                if t in keyify(c):
                    found_this_t = True
                    break
            if found_this_t:
                hits_by_target[t].append((csv, header_line))

    print("\n" + "=" * 100)
    print("RESULTAAT: PER ZOEKWOORD")
    print("=" * 100)

    for original in TARGET_COLS:
        t = keyify(original)

        print("\n" + "=" * 30)
        print(original)
        print("-" * 30)

        csv_hits = hits_by_target.get(t, [])
        if not csv_hits:
            print("bestandsnaam:")
            print("  (geen CSV gevonden met deze kolomnaam in de header)")
            print("-" * 24)
            print("python:")
            print("  (overgeslagen: geen relevante CSV’s om op te zoeken)")
            continue

        # 1) CSV output
        print("bestandsnaam:")
        # verzamel stems van de CSV’s voor python-zoek later
        stems: Set[str] = set()

        for csv, header_line in sorted(csv_hits, key=lambda x: str(x[0])):
            stems.add(csv.stem)
            print(f"  {short_loc(csv, DATA_ROOT)}")
            print(f"  header: {header_line}")

        # 2) Python output
        print("-" * 24)
        print("python:")

        # zoek in python op:
        # - elke csv stem (beste match met jouw oude stap 2)
        # - óók het zoekwoord zelf (kolomnaam hardcoded)
        py_hits_all: List[Tuple[str, PyHit]] = []

        for stem in sorted(stems):
            for h in find_in_python(stem, py_files):
                py_hits_all.append((f"CSV-stem '{stem}'", h))

        for h in find_in_python(original, py_files):
            py_hits_all.append((f"kolomnaam '{original}'", h))

        if not py_hits_all:
            print("  (geen hits in python code op csv-stem(s) of op het zoekwoord zelf)")
        else:
            # dedupe
            seen = set()
            for why, h in sorted(py_hits_all, key=lambda x: (str(x[1].py_file), x[1].line_no, x[0])):
                key = (h.py_file, h.line_no, h.line)
                if key in seen:
                    continue
                seen.add(key)
                print(f"  {h.py_file.name} | {h.py_file.parent.name}")
                print(f"  L{h.line_no}: {h.line}   ({why})")

    if unreadable:
        print("\n" + "-" * 100)
        print(f"LET OP: {len(unreadable)} CSV files kon ik niet lezen.")
        print("Eerste 10:")
        for f in unreadable[:10]:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
