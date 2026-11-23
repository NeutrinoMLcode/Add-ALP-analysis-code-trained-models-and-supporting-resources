#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_delphes_llp_boundaries.py

Purpose
-------
Scan many Delphes ROOT files and estimate the effective displacement region
where Delphes-LLP reconstructs LLP candidates.

For each ROOT file, this script:
  1. Reads GenParticle truth information.
  2. Finds ALP particles by PDG ID.
  3. Uses the first daughter vertex as the ALP decay vertex.
  4. Computes:
       R_cm    = sqrt(x^2 + y^2) / 10
       L3D_cm  = sqrt(x^2 + y^2 + z^2) / 10
     assuming Delphes Particle.X/Y/Z are in mm.
  5. Reads LLP_size or LLPs_size.
  6. Estimates the displacement range of reconstructed LLP candidates
     using events with LLP_size > 0.
  7. Writes:
       - one summary CSV over all ROOT files;
       - optionally one per-event CSV.

Important
---------
This does NOT prove that Delphes has a hard LLP decay-length window.
It estimates the effective reconstructed region from events with LLP_size > 0.
If the Delphes card has no explicit 1--200 cm cut, the observed boundary comes
from detector geometry, track/vertex requirements, and LLPReconstruction cuts.

Usage
-----
Example:
    python scan_delphes_llp_boundaries.py ^
        --root-dir "D:/share/ALP/Sensitivity/codex/root_files" ^
        --out-dir  "D:/share/ALP/Sensitivity/codex/llp_boundary_check" ^
        --alp-id 9000001 ^
        --save-events

Minimal:
    python scan_delphes_llp_boundaries.py --root-dir "." --out-dir "./llp_boundary_check"

Dependencies
------------
    pip install uproot awkward pandas numpy
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import awkward as ak
import numpy as np
import pandas as pd
import uproot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Delphes ROOT files and estimate LLP displacement boundaries."
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Directory containing ROOT files. The script searches recursively.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for CSV results.",
    )
    parser.add_argument(
        "--tree-name",
        default="Delphes",
        help="TTree name inside ROOT files. Default: Delphes.",
    )
    parser.add_argument(
        "--alp-id",
        type=int,
        default=9000001,
        help="PDG ID of ALP. Default: 9000001.",
    )
    parser.add_argument(
        "--pattern",
        default="*.root",
        help="ROOT filename pattern. Default: *.root.",
    )
    parser.add_argument(
        "--save-events",
        action="store_true",
        help="Save per-event CSV for all files. Can be large.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional maximum number of events per ROOT file.",
    )
    parser.add_argument(
        "--llp-window-min-cm",
        type=float,
        default=1.0,
        help="Reference lower boundary in cm for diagnostic fractions. Default: 1 cm.",
    )
    parser.add_argument(
        "--llp-window-max-cm",
        type=float,
        default=200.0,
        help="Reference upper boundary in cm for diagnostic fractions. Default: 200 cm.",
    )
    return parser.parse_args()


def safe_quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, q))


def safe_min(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.min(values))


def safe_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.max(values))


def find_llp_size_branch(branches: Sequence[str]) -> Optional[str]:
    """
    Common branch names:
      LLP_size
      LLPs_size
      LLP.LLP_size-like variants are unlikely in Delphes flat tree,
      but we keep a loose fallback.
    """
    branch_set = set(branches)
    if "LLP_size" in branch_set:
        return "LLP_size"
    if "LLPs_size" in branch_set:
        return "LLPs_size"

    candidates = [b for b in branches if b.lower() in {"llp_size", "llps_size"}]
    if candidates:
        return candidates[0]

    candidates = [
        b for b in branches
        if ("llp" in b.lower() and b.lower().endswith("_size"))
    ]
    if candidates:
        # Prefer the shortest/simple name.
        candidates = sorted(candidates, key=len)
        return candidates[0]

    return None


def extract_ma_fa_from_name(path: Path) -> Tuple[float, float]:
    """
    Try to extract ma and fa/Lambda from filenames such as:
      1.4geV-4000GeV.root
      ma_1p4GeV_fa_4TeV.root
      alp_ma_1p4GeV_fa_4TeV.root

    Returns (nan, nan) if parsing fails.
    fa is returned in GeV.
    """
    name = path.stem

    # Pattern: 1.4geV-4000GeV
    m = re.search(
        r"(?P<ma>\d+(?:\.\d+)?|(?:\d+p\d+))\s*gev\s*[-_]\s*(?P<fa>\d+(?:\.\d+)?|(?:\d+p\d+))\s*(?P<unit>gev|tev)",
        name,
        flags=re.IGNORECASE,
    )
    if m:
        ma = float(m.group("ma").replace("p", "."))
        fa = float(m.group("fa").replace("p", "."))
        if m.group("unit").lower() == "tev":
            fa *= 1000.0
        return ma, fa

    # Pattern: ma_1p4GeV_fa_4TeV
    m_ma = re.search(r"ma[_-]?(?P<ma>\d+(?:\.\d+)?|(?:\d+p\d+))\s*gev", name, flags=re.IGNORECASE)
    m_fa = re.search(r"(fa|lambda)[_-]?(?P<fa>\d+(?:\.\d+)?|(?:\d+p\d+))\s*(?P<unit>gev|tev)", name, flags=re.IGNORECASE)
    if m_ma and m_fa:
        ma = float(m_ma.group("ma").replace("p", "."))
        fa = float(m_fa.group("fa").replace("p", "."))
        if m_fa.group("unit").lower() == "tev":
            fa *= 1000.0
        return ma, fa

    return float("nan"), float("nan")


def read_required_arrays(
    root_path: Path,
    tree_name: str,
    max_events: Optional[int],
) -> Tuple[Optional[ak.Array], Optional[str], List[str], Optional[str]]:
    """
    Returns:
      arrays, llp_size_branch, branch_names, error_message
    """
    try:
        with uproot.open(root_path) as f:
            if tree_name not in f:
                return None, None, [], f"Tree '{tree_name}' not found. Available keys: {list(f.keys())[:10]}"

            tree = f[tree_name]
            branches = list(tree.keys())
            llp_size_branch = find_llp_size_branch(branches)

            needed = [
                "Particle.PID",
                "Particle.D1",
                "Particle.D2",
                "Particle.X",
                "Particle.Y",
                "Particle.Z",
            ]

            missing = [b for b in needed if b not in branches]
            if missing:
                return None, llp_size_branch, branches, f"Missing required branches: {missing}"

            read_branches = needed.copy()
            if llp_size_branch is not None:
                read_branches.append(llp_size_branch)

            entry_stop = max_events if max_events is not None else None
            arrays = tree.arrays(read_branches, library="ak", entry_stop=entry_stop)

            return arrays, llp_size_branch, branches, None

    except Exception as exc:
        return None, None, [], repr(exc)


def extract_events_from_file(
    root_path: Path,
    tree_name: str,
    alp_id: int,
    max_events: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    arrays, llp_size_branch, branches, error = read_required_arrays(root_path, tree_name, max_events)

    ma_name, fa_name = extract_ma_fa_from_name(root_path)

    base_info: Dict[str, object] = {
        "file": str(root_path),
        "file_name": root_path.name,
        "ma_from_filename_GeV": ma_name,
        "fa_from_filename_GeV": fa_name,
        "llp_size_branch": llp_size_branch if llp_size_branch is not None else "",
        "read_error": error if error is not None else "",
    }

    if error is not None or arrays is None:
        empty = pd.DataFrame()
        return empty, base_info

    pid = arrays["Particle.PID"]
    d1 = arrays["Particle.D1"]
    d2 = arrays["Particle.D2"]
    x = arrays["Particle.X"]
    y = arrays["Particle.Y"]
    z = arrays["Particle.Z"]

    n_events = len(pid)
    records: List[Dict[str, object]] = []

    for iev in range(n_events):
        pid_evt = np.asarray(pid[iev])
        d1_evt = np.asarray(d1[iev])
        d2_evt = np.asarray(d2[iev])
        x_evt = np.asarray(x[iev])
        y_evt = np.asarray(y[iev])
        z_evt = np.asarray(z[iev])

        if llp_size_branch is None:
            llp_size = np.nan
        else:
            try:
                llp_size = int(arrays[llp_size_branch][iev])
            except Exception:
                llp_size = np.nan

        alp_indices = np.where(np.abs(pid_evt) == alp_id)[0]

        if len(alp_indices) == 0:
            records.append(
                {
                    "file_name": root_path.name,
                    "event": iev,
                    "has_alp": False,
                    "has_decay_vertex": False,
                    "n_alp": 0,
                    "alp_index": np.nan,
                    "d1_index": np.nan,
                    "d2_index": np.nan,
                    "d1_pid": np.nan,
                    "d2_pid": np.nan,
                    "R_cm": np.nan,
                    "L3D_cm": np.nan,
                    "LLP_size": llp_size,
                    "has_reco_llp": bool(llp_size > 0) if np.isfinite(llp_size) else False,
                }
            )
            continue

        # If there are multiple ALPs, use the first ALP.
        # For most e+e- -> gamma ALP samples, there should be exactly one.
        ia = int(alp_indices[0])
        id1 = int(d1_evt[ia])
        id2 = int(d2_evt[ia])

        if id1 < 0 or id1 >= len(pid_evt):
            records.append(
                {
                    "file_name": root_path.name,
                    "event": iev,
                    "has_alp": True,
                    "has_decay_vertex": False,
                    "n_alp": int(len(alp_indices)),
                    "alp_index": ia,
                    "d1_index": id1,
                    "d2_index": id2,
                    "d1_pid": np.nan,
                    "d2_pid": np.nan,
                    "R_cm": np.nan,
                    "L3D_cm": np.nan,
                    "LLP_size": llp_size,
                    "has_reco_llp": bool(llp_size > 0) if np.isfinite(llp_size) else False,
                }
            )
            continue

        xd = float(x_evt[id1])
        yd = float(y_evt[id1])
        zd = float(z_evt[id1])

        # Delphes Particle.X/Y/Z are usually in mm.
        R_cm = math.sqrt(xd * xd + yd * yd) / 10.0
        L3D_cm = math.sqrt(xd * xd + yd * yd + zd * zd) / 10.0

        d1_pid = int(pid_evt[id1])
        d2_pid = int(pid_evt[id2]) if 0 <= id2 < len(pid_evt) else np.nan

        records.append(
            {
                "file_name": root_path.name,
                "event": iev,
                "has_alp": True,
                "has_decay_vertex": True,
                "n_alp": int(len(alp_indices)),
                "alp_index": ia,
                "d1_index": id1,
                "d2_index": id2,
                "d1_pid": d1_pid,
                "d2_pid": d2_pid,
                "R_cm": R_cm,
                "L3D_cm": L3D_cm,
                "LLP_size": llp_size,
                "has_reco_llp": bool(llp_size > 0) if np.isfinite(llp_size) else False,
            }
        )

    df = pd.DataFrame(records)
    return df, base_info


def summarize_file(
    df: pd.DataFrame,
    base_info: Dict[str, object],
    win_min_cm: float,
    win_max_cm: float,
) -> Dict[str, object]:
    out = dict(base_info)

    if df.empty:
        out.update(
            {
                "N_total": 0,
                "N_has_ALP": 0,
                "N_has_decay_vertex": 0,
                "N_LLP_size_gt_0": 0,
                "eff_reco_LLP_size_gt_0": np.nan,
            }
        )
        return out

    n_total = len(df)
    valid = df["has_decay_vertex"].astype(bool)
    has_alp = df["has_alp"].astype(bool)
    has_reco = df["has_reco_llp"].astype(bool)

    truth_L3D_win = valid & (df["L3D_cm"] > win_min_cm) & (df["L3D_cm"] < win_max_cm)
    truth_R_win = valid & (df["R_cm"] > win_min_cm) & (df["R_cm"] < win_max_cm)

    reco_df = df[has_reco & valid].copy()

    # Values for all truth decays
    R_all = df.loc[valid, "R_cm"].to_numpy(dtype=float)
    L_all = df.loc[valid, "L3D_cm"].to_numpy(dtype=float)

    # Values for reconstructed LLP candidates
    R_reco = reco_df["R_cm"].to_numpy(dtype=float)
    L_reco = reco_df["L3D_cm"].to_numpy(dtype=float)

    out.update(
        {
            "N_total": int(n_total),
            "N_has_ALP": int(has_alp.sum()),
            "N_has_decay_vertex": int(valid.sum()),
            "N_LLP_size_gt_0": int(has_reco.sum()),
            "eff_reco_LLP_size_gt_0": float(has_reco.sum() / n_total) if n_total else np.nan,

            f"N_truth_L3D_{win_min_cm:g}_{win_max_cm:g}cm": int(truth_L3D_win.sum()),
            f"P_truth_L3D_{win_min_cm:g}_{win_max_cm:g}cm": float(truth_L3D_win.sum() / n_total) if n_total else np.nan,
            f"N_truth_R_{win_min_cm:g}_{win_max_cm:g}cm": int(truth_R_win.sum()),
            f"P_truth_R_{win_min_cm:g}_{win_max_cm:g}cm": float(truth_R_win.sum() / n_total) if n_total else np.nan,

            f"N_reco_inside_L3D_{win_min_cm:g}_{win_max_cm:g}cm": int((has_reco & truth_L3D_win).sum()),
            f"N_reco_outside_L3D_{win_min_cm:g}_{win_max_cm:g}cm": int((has_reco & ~truth_L3D_win).sum()),
            f"N_reco_inside_R_{win_min_cm:g}_{win_max_cm:g}cm": int((has_reco & truth_R_win).sum()),
            f"N_reco_outside_R_{win_min_cm:g}_{win_max_cm:g}cm": int((has_reco & ~truth_R_win).sum()),

            # All truth decay distributions
            "R_all_min_cm": safe_min(R_all),
            "R_all_q01_cm": safe_quantile(R_all, 0.01),
            "R_all_q05_cm": safe_quantile(R_all, 0.05),
            "R_all_q50_cm": safe_quantile(R_all, 0.50),
            "R_all_q95_cm": safe_quantile(R_all, 0.95),
            "R_all_q99_cm": safe_quantile(R_all, 0.99),
            "R_all_max_cm": safe_max(R_all),

            "L3D_all_min_cm": safe_min(L_all),
            "L3D_all_q01_cm": safe_quantile(L_all, 0.01),
            "L3D_all_q05_cm": safe_quantile(L_all, 0.05),
            "L3D_all_q50_cm": safe_quantile(L_all, 0.50),
            "L3D_all_q95_cm": safe_quantile(L_all, 0.95),
            "L3D_all_q99_cm": safe_quantile(L_all, 0.99),
            "L3D_all_max_cm": safe_max(L_all),

            # Reconstructed LLP candidate displacement distributions.
            "R_reco_min_cm": safe_min(R_reco),
            "R_reco_q01_cm": safe_quantile(R_reco, 0.01),
            "R_reco_q05_cm": safe_quantile(R_reco, 0.05),
            "R_reco_q50_cm": safe_quantile(R_reco, 0.50),
            "R_reco_q95_cm": safe_quantile(R_reco, 0.95),
            "R_reco_q99_cm": safe_quantile(R_reco, 0.99),
            "R_reco_max_cm": safe_max(R_reco),

            "L3D_reco_min_cm": safe_min(L_reco),
            "L3D_reco_q01_cm": safe_quantile(L_reco, 0.01),
            "L3D_reco_q05_cm": safe_quantile(L_reco, 0.05),
            "L3D_reco_q50_cm": safe_quantile(L_reco, 0.50),
            "L3D_reco_q95_cm": safe_quantile(L_reco, 0.95),
            "L3D_reco_q99_cm": safe_quantile(L_reco, 0.99),
            "L3D_reco_max_cm": safe_max(L_reco),
        }
    )

    return out


def summarize_global(all_events: pd.DataFrame, win_min_cm: float, win_max_cm: float) -> Dict[str, object]:
    if all_events.empty:
        return {"N_total": 0}

    n_total = len(all_events)
    valid = all_events["has_decay_vertex"].astype(bool)
    has_reco = all_events["has_reco_llp"].astype(bool)

    truth_L3D_win = valid & (all_events["L3D_cm"] > win_min_cm) & (all_events["L3D_cm"] < win_max_cm)
    truth_R_win = valid & (all_events["R_cm"] > win_min_cm) & (all_events["R_cm"] < win_max_cm)

    reco = all_events[has_reco & valid]

    R_reco = reco["R_cm"].to_numpy(dtype=float)
    L_reco = reco["L3D_cm"].to_numpy(dtype=float)

    return {
        "N_total_events_all_files": int(n_total),
        "N_files": int(all_events["file_name"].nunique()),
        "N_reco_LLP_size_gt_0": int(has_reco.sum()),
        "global_eff_reco": float(has_reco.sum() / n_total) if n_total else np.nan,
        f"global_N_truth_L3D_{win_min_cm:g}_{win_max_cm:g}cm": int(truth_L3D_win.sum()),
        f"global_P_truth_L3D_{win_min_cm:g}_{win_max_cm:g}cm": float(truth_L3D_win.sum() / n_total) if n_total else np.nan,
        f"global_N_truth_R_{win_min_cm:g}_{win_max_cm:g}cm": int(truth_R_win.sum()),
        f"global_P_truth_R_{win_min_cm:g}_{win_max_cm:g}cm": float(truth_R_win.sum() / n_total) if n_total else np.nan,
        f"global_N_reco_inside_L3D_{win_min_cm:g}_{win_max_cm:g}cm": int((has_reco & truth_L3D_win).sum()),
        f"global_N_reco_outside_L3D_{win_min_cm:g}_{win_max_cm:g}cm": int((has_reco & ~truth_L3D_win).sum()),
        "global_R_reco_min_cm": safe_min(R_reco),
        "global_R_reco_q01_cm": safe_quantile(R_reco, 0.01),
        "global_R_reco_q05_cm": safe_quantile(R_reco, 0.05),
        "global_R_reco_q50_cm": safe_quantile(R_reco, 0.50),
        "global_R_reco_q95_cm": safe_quantile(R_reco, 0.95),
        "global_R_reco_q99_cm": safe_quantile(R_reco, 0.99),
        "global_R_reco_max_cm": safe_max(R_reco),
        "global_L3D_reco_min_cm": safe_min(L_reco),
        "global_L3D_reco_q01_cm": safe_quantile(L_reco, 0.01),
        "global_L3D_reco_q05_cm": safe_quantile(L_reco, 0.05),
        "global_L3D_reco_q50_cm": safe_quantile(L_reco, 0.50),
        "global_L3D_reco_q95_cm": safe_quantile(L_reco, 0.95),
        "global_L3D_reco_q99_cm": safe_quantile(L_reco, 0.99),
        "global_L3D_reco_max_cm": safe_max(L_reco),
    }


def main() -> None:
    args = parse_args()

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root_files = sorted(root_dir.rglob(args.pattern))
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found under {root_dir} with pattern {args.pattern}")

    summaries: List[Dict[str, object]] = []
    event_frames: List[pd.DataFrame] = []

    for i, root_path in enumerate(root_files, start=1):
        print(f"[{i}/{len(root_files)}] Processing: {root_path}")

        df_events, base_info = extract_events_from_file(
            root_path=root_path,
            tree_name=args.tree_name,
            alp_id=args.alp_id,
            max_events=args.max_events,
        )

        summary = summarize_file(
            df=df_events,
            base_info=base_info,
            win_min_cm=args.llp_window_min_cm,
            win_max_cm=args.llp_window_max_cm,
        )
        summaries.append(summary)

        if args.save_events and not df_events.empty:
            # Add absolute file path only in saved event table.
            df_events.insert(0, "file", str(root_path))
            event_frames.append(df_events)

    summary_df = pd.DataFrame(summaries)
    summary_path = out_dir / "delphes_llp_boundary_summary_by_file.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved summary: {summary_path}")

    if args.save_events:
        if event_frames:
            all_events = pd.concat(event_frames, ignore_index=True)
        else:
            all_events = pd.DataFrame()

        events_path = out_dir / "delphes_llp_boundary_events.csv"
        all_events.to_csv(events_path, index=False)
        print(f"Saved events:  {events_path}")
    else:
        all_events = pd.DataFrame()

    # For global summary, if events were not saved, build a lightweight global
    # table by re-reading only reconstructed-event info would be more complicated.
    # Therefore only produce global event-level quantiles when --save-events is used.
    if args.save_events and not all_events.empty:
        global_summary = summarize_global(
            all_events=all_events,
            win_min_cm=args.llp_window_min_cm,
            win_max_cm=args.llp_window_max_cm,
        )
        global_path = out_dir / "delphes_llp_boundary_global_summary.csv"
        pd.DataFrame([global_summary]).to_csv(global_path, index=False)
        print(f"Saved global:  {global_path}")

        print("\nGlobal reconstructed-LLP displacement estimate:")
        print(pd.DataFrame([global_summary]).T)

    print("\nNotes:")
    print("  - R_reco_* and L3D_reco_* are computed only for events with LLP_size > 0.")
    print("  - Use q01/q99 rather than min/max as a more stable estimate of the effective region.")
    print("  - If q01 is below 1 cm, then Delphes-LLP is reconstructing candidates below your 1 cm LLP window.")
    print("  - If q99 or max is far below 200 cm, that does not prove a hard upper cut; it may be caused by event lifetimes.")


if __name__ == "__main__":
    main()
