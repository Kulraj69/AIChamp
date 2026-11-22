#!/usr/bin/env python3
"""Validate an Enveda CASMI26 submission.csv before spending a Kaggle slot."""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def find_test_parquet(data_dir: str | None) -> Path | None:
    candidates = []
    if data_dir:
        candidates.append(Path(data_dir) / "test.parquet")
    candidates += [
        Path("/home/ubuntu/repos/enveda-casmi26/data/test.parquet"),
        Path("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra/test.parquet"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    hits = glob.glob("/kaggle/input/**/test.parquet", recursive=True)
    return Path(sorted(hits, key=len)[0]) if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=Path)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--skip-keys", action="store_true", help="skip RDKit InChIKey14 uniqueness check")
    ap.add_argument("--skip-chemistry", action="store_true", help="skip RDKit SMILES/InChIKey checks")
    args = ap.parse_args()

    df = pd.read_csv(args.submission, dtype=str, keep_default_na=False)
    errors = []
    if list(df.columns) != ["molecule_id", "smiles"]:
        errors.append(f"columns must be ['molecule_id', 'smiles'], got {list(df.columns)}")
        print("\n".join(errors))
        return 1
    if df.empty:
        errors.append("submission is empty")
        print("\n".join(errors))
        return 1
    if not df["molecule_id"].is_unique:
        errors.append("molecule_id is not unique")
    if (df["smiles"].str.len() == 0).any():
        errors.append("empty smiles field")
    candidate_lists = df["smiles"].str.split(";")
    counts = candidate_lists.map(len)
    empty_candidates = candidate_lists.map(lambda parts: sum(not part.strip() for part in parts)).sum()
    if empty_candidates:
        errors.append(f"{int(empty_candidates)} empty candidate SMILES")
    if counts.max() > 25:
        errors.append(f"max guesses {counts.max()} > 25")
    if counts.min() < 1:
        errors.append("at least one molecule has no guesses")

    test_path = find_test_parquet(args.data_dir)
    if test_path is not None:
        test_mols = set(pq.read_table(test_path, columns=["molecule_id"]).to_pandas()["molecule_id"].astype(str))
        got = set(df["molecule_id"].astype(str))
        if got != test_mols:
            errors.append(f"molecule_id set mismatch: missing {len(test_mols - got)}, extra {len(got - test_mols)}")
        print(f"test_parquet={test_path} molecules={len(test_mols)} submission_rows={len(df)}")
    else:
        print("warning: test.parquet not found; molecule_id coverage not checked")

    exact_unique = df["smiles"].str.split(";").map(lambda parts: len(set(parts)))
    print(f"rows={len(df)} guesses min={counts.min()} median={int(counts.median())} max={counts.max()}")
    print(f"exact unique SMILES per molecule min={exact_unique.min()} median={int(exact_unique.median())} mean={exact_unique.mean():.2f}")

    if not args.skip_chemistry:
        try:
            from rdkit import Chem
            from rdkit.Chem import inchi
        except Exception as exc:
            print(f"warning: RDKit unavailable; chemistry checks skipped ({exc})")
        else:
            invalid_candidates = 0
            missing_keys = 0
            key_counts = []
            for smiles in df["smiles"]:
                keys = set()
                for candidate in smiles.split(";"):
                    if not candidate.strip():
                        continue
                    try:
                        mol = Chem.MolFromSmiles(candidate)
                    except Exception:
                        mol = None
                    if mol is None:
                        invalid_candidates += 1
                        continue
                    try:
                        key = inchi.MolToInchiKey(mol)
                    except Exception:
                        key = None
                    if key:
                        keys.add(key[:14])
                    else:
                        missing_keys += 1
                key_counts.append(len(keys))
            if invalid_candidates:
                errors.append(f"{invalid_candidates} candidate SMILES are not parseable by RDKit")
            if missing_keys:
                errors.append(f"{missing_keys} candidate SMILES have no InChIKey14")
            if not args.skip_keys:
                key_counts_arr = np.asarray(key_counts)
                print(f"InChIKey14 unique per molecule min={key_counts_arr.min()} median={int(np.median(key_counts_arr))} mean={key_counts_arr.mean():.2f} count<25={int((key_counts_arr < 25).sum())}")

    if errors:
        print("FAIL")
        for e in errors:
            print(" -", e)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
