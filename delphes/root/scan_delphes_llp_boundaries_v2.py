#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_delphes_llp_boundaries_v2.py

Robust no-awkward scanner for Delphes ROOT files.

Why v2?
-------
Some Delphes ROOT files expose branches as e.g.
  Particle/Particle.PID
rather than
  Particle.PID
The previous script only looked for exact names, so it could fill NaN everywhere.
This version resolves branch names by suffix and can also infer LLP_size from
LLP candidate subbranches if an explicit LLP_size branch is absent.

Dependencies:
  uproot, numpy

Example:
  python scan_delphes_llp_boundaries_v2.py \
    --root-dir "/home/wsy/MG5_aMC_v3_5_6_LLP/ALP_LLP" \
    --out-dir "/home/wsy/MG5_aMC_v3_5_6_LLP/ALP_LLP/llp_boundary_check_v2" \
    --alp-id 9000001 \
    --pattern "tag_1_delphes_events.root"

Optional debug for first few files:
  --print-branches 3
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import uproot


PARTICLE_COLLECTION_CANDIDATES = [
    "Particle",
    "GenParticle",
    "GenParticles",
    "TruthParticle",
    "TruthParticles",
]

LLP_COLLECTION_CANDIDATES = [
    "LLPs",
    "LLP",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tree-name", default="Delphes")
    p.add_argument("--alp-id", type=int, default=9000001)
    p.add_argument("--pattern", default="*.root")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--save-events", action="store_true")
    p.add_argument("--llp-window-min-cm", type=float, default=1.0)
    p.add_argument("--llp-window-max-cm", type=float, default=200.0)
    p.add_argument(
        "--print-branches",
        type=int,
        default=0,
        help="Print available branches for the first N files, useful for debugging.",
    )
    return p.parse_args()


def parse_ma_fa_from_path(path: Path) -> Tuple[float, float]:
    text = str(path)
    m_ma = re.search(r"ma[_-]?(?P<ma>\d+(?:\.\d+)?|\d+p\d+)\s*gev", text, re.I)
    m_fa = re.search(r"(fa|lambda)[_-]?(?P<fa>\d+(?:\.\d+)?|\d+p\d+)\s*(?P<unit>gev|tev)", text, re.I)
    if m_ma and m_fa:
        ma = float(m_ma.group("ma").replace("p", "."))
        fa = float(m_fa.group("fa").replace("p", "."))
        if m_fa.group("unit").lower() == "tev":
            fa *= 1000.0
        return ma, fa
    return float("nan"), float("nan")


def get_all_keys(tree) -> List[str]:
    # uproot versions differ; recursive=True is preferred when available.
    try:
        keys = list(tree.keys(recursive=True))
    except TypeError:
        keys = list(tree.keys())
    return [str(k) for k in keys]


def leaf_name(branch: str) -> str:
    # Particle/Particle.PID -> Particle.PID
    return branch.split("/")[-1]


def find_field_branch(branches: Sequence[str], collection: str, field: str) -> Optional[str]:
    """Find a branch for collection.field, allowing slash paths."""
    exact_candidates = [
        f"{collection}.{field}",
        f"{collection}/{collection}.{field}",
    ]
    bset = set(branches)
    for cand in exact_candidates:
        if cand in bset:
            return cand

    target = f"{collection}.{field}"
    for b in branches:
        if leaf_name(b) == target:
            return b

    # Loose fallback: contains collection and leaf ends with .field.
    # This catches cases like Delphes/Particle/Particle.PID.
    for b in branches:
        lf = leaf_name(b)
        if collection.lower() in b.lower() and lf.lower().endswith(f".{field.lower()}"):
            return b

    return None


def resolve_particle_branches(branches: Sequence[str]) -> Tuple[Optional[str], Dict[str, str]]:
    required_fields = ["PID", "D1", "D2", "X", "Y", "Z"]
    for coll in PARTICLE_COLLECTION_CANDIDATES:
        mapping = {}
        for field in required_fields:
            b = find_field_branch(branches, coll, field)
            if b is not None:
                mapping[field] = b
        if len(mapping) == len(required_fields):
            return coll, mapping
    return None, {}


def find_llp_size_branch(branches: Sequence[str]) -> Optional[str]:
    bset = set(branches)
    exact = ["LLP_size", "LLPs_size", "LLP/LLP_size", "LLPs/LLPs_size"]
    for cand in exact:
        if cand in bset:
            return cand
    for b in branches:
        lf = leaf_name(b).lower()
        if "llp" in b.lower() and lf.endswith("_size"):
            return b
    return None


def find_llp_count_field(branches: Sequence[str]) -> Optional[str]:
    """If no LLP_size exists, use any LLP candidate subbranch to count candidates."""
    preferred_fields = ["PT", "Eta", "Phi", "Mass", "X", "Y", "Z"]
    for coll in LLP_COLLECTION_CANDIDATES:
        for field in preferred_fields:
            b = find_field_branch(branches, coll, field)
            if b is not None:
                return b
    # Loose fallback: any branch whose leaf is LLPs.something / LLP.something.
    for b in branches:
        lf = leaf_name(b).lower()
        if (lf.startswith("llps.") or lf.startswith("llp.")) and not lf.endswith("_size"):
            return b
    return None


def finite_array(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def quant(values, prob):
    arr = finite_array(values)
    if len(arr) == 0:
        return float("nan")
    return float(np.quantile(arr, prob))


def amin(values):
    arr = finite_array(values)
    return float(np.min(arr)) if len(arr) else float("nan")


def amax(values):
    arr = finite_array(values)
    return float(np.max(arr)) if len(arr) else float("nan")


def to_event_array(x):
    return np.asarray(x)


def event_count_from_branch_entry(x) -> int:
    arr = np.asarray(x)
    if arr.shape == ():
        try:
            return int(arr.item())
        except Exception:
            return int(arr)
    return int(len(arr))


def scalar_from_entry(x):
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    if len(arr) == 1:
        y = arr[0]
        return y.item() if hasattr(y, "item") else y
    return arr.item()


def read_root_arrays(root_path: Path, tree_name: str, max_events: Optional[int], print_branches: bool):
    with uproot.open(root_path) as f:
        if tree_name not in f:
            raise RuntimeError(f"Cannot find tree {tree_name}. Keys = {list(f.keys())[:20]}")
        tree = f[tree_name]
        branches = get_all_keys(tree)

        if print_branches:
            print(f"\n--- Branches in {root_path} ---")
            for b in branches[:300]:
                print(b)
            if len(branches) > 300:
                print(f"... ({len(branches)-300} more branches not printed)")
            print("--- end branches ---\n")

        particle_collection, pmap = resolve_particle_branches(branches)
        if not pmap:
            short = ", ".join(branches[:80])
            raise RuntimeError(
                "Could not resolve truth Particle branches for PID/D1/D2/X/Y/Z. "
                f"First branches: {short}"
            )

        llp_size_branch = find_llp_size_branch(branches)
        llp_count_field = None
        if llp_size_branch is None:
            llp_count_field = find_llp_count_field(branches)

        read_branches = list(pmap.values())
        if llp_size_branch is not None:
            read_branches.append(llp_size_branch)
        elif llp_count_field is not None:
            read_branches.append(llp_count_field)

        arrays = tree.arrays(read_branches, library="np", entry_stop=max_events)

    return arrays, particle_collection, pmap, llp_size_branch, llp_count_field


def empty_summary(root_path: Path, error_msg: str) -> Dict[str, object]:
    ma, fa = parse_ma_fa_from_path(root_path)
    row = {
        "file": str(root_path),
        "file_name": root_path.name,
        "ma_from_path_GeV": ma,
        "fa_from_path_GeV": fa,
        "particle_collection": "",
        "llp_size_branch": "",
        "llp_count_field": "",
        "read_error": error_msg,
    }
    for k in SUMMARY_NUMERIC_KEYS:
        row[k] = float("nan")
    return row


SUMMARY_NUMERIC_KEYS = [
    "N_total", "N_has_ALP", "N_has_decay_vertex",
    "N_LLP_size_gt_0", "eff_reco_LLP_size_gt_0",
    "N_truth_L3D_window", "P_truth_L3D_window",
    "N_truth_R_window", "P_truth_R_window",
    "N_reco_inside_L3D_window", "N_reco_outside_L3D_window",
    "N_reco_inside_R_window", "N_reco_outside_R_window",
    "R_all_min_cm", "R_all_q01_cm", "R_all_q05_cm", "R_all_q50_cm",
    "R_all_q95_cm", "R_all_q99_cm", "R_all_max_cm",
    "L3D_all_min_cm", "L3D_all_q01_cm", "L3D_all_q05_cm", "L3D_all_q50_cm",
    "L3D_all_q95_cm", "L3D_all_q99_cm", "L3D_all_max_cm",
    "R_reco_min_cm", "R_reco_q01_cm", "R_reco_q05_cm", "R_reco_q50_cm",
    "R_reco_q95_cm", "R_reco_q99_cm", "R_reco_max_cm",
    "L3D_reco_min_cm", "L3D_reco_q01_cm", "L3D_reco_q05_cm", "L3D_reco_q50_cm",
    "L3D_reco_q95_cm", "L3D_reco_q99_cm", "L3D_reco_max_cm",
]


def process_file(root_path, tree_name, alp_id, max_events, win_min, win_max, save_events, print_branches=False):
    ma, fa = parse_ma_fa_from_path(root_path)
    try:
        arrays, particle_collection, pmap, llp_size_branch, llp_count_field = read_root_arrays(
            root_path, tree_name, max_events, print_branches
        )
    except Exception as e:
        return empty_summary(root_path, repr(e)), []

    pid_all = arrays[pmap["PID"]]
    d1_all = arrays[pmap["D1"]]
    d2_all = arrays[pmap["D2"]]
    x_all = arrays[pmap["X"]]
    y_all = arrays[pmap["Y"]]
    z_all = arrays[pmap["Z"]]

    llp_arr = None
    if llp_size_branch is not None:
        llp_arr = arrays[llp_size_branch]
    elif llp_count_field is not None:
        llp_arr = arrays[llp_count_field]

    n_total = len(pid_all)
    n_has_alp = 0
    n_has_decay_vertex = 0
    n_reco = 0
    n_truth_l3d_win = 0
    n_truth_r_win = 0
    n_reco_inside_l3d = 0
    n_reco_outside_l3d = 0
    n_reco_inside_r = 0
    n_reco_outside_r = 0

    R_all, L_all, R_reco, L_reco = [], [], [], []
    event_rows = []

    for iev in range(n_total):
        pid_evt = to_event_array(pid_all[iev])
        d1_evt = to_event_array(d1_all[iev])
        d2_evt = to_event_array(d2_all[iev])
        x_evt = to_event_array(x_all[iev])
        y_evt = to_event_array(y_all[iev])
        z_evt = to_event_array(z_all[iev])

        if llp_arr is None:
            llp_size = float("nan")
            has_reco = False
        else:
            try:
                if llp_size_branch is not None:
                    llp_size = int(scalar_from_entry(llp_arr[iev]))
                else:
                    llp_size = event_count_from_branch_entry(llp_arr[iev])
                has_reco = bool(llp_size > 0)
            except Exception:
                llp_size = float("nan")
                has_reco = False

        if has_reco:
            n_reco += 1

        alp_indices = np.where(np.abs(pid_evt) == alp_id)[0]

        has_alp = len(alp_indices) > 0
        has_decay_vertex = False
        R_cm = float("nan")
        L3D_cm = float("nan")
        d1_pid = ""
        d2_pid = ""
        ia = ""
        id1 = ""
        id2 = ""
        inside_l3d = False
        inside_r = False

        if has_alp:
            n_has_alp += 1
            ia = int(alp_indices[0])
            id1 = int(d1_evt[ia])
            id2 = int(d2_evt[ia])

            if 0 <= id1 < len(pid_evt):
                has_decay_vertex = True
                n_has_decay_vertex += 1

                xd = float(x_evt[id1])
                yd = float(y_evt[id1])
                zd = float(z_evt[id1])

                # Delphes Particle.X/Y/Z are usually in mm.
                R_cm = math.sqrt(xd * xd + yd * yd) / 10.0
                L3D_cm = math.sqrt(xd * xd + yd * yd + zd * zd) / 10.0

                d1_pid = int(pid_evt[id1])
                if 0 <= id2 < len(pid_evt):
                    d2_pid = int(pid_evt[id2])

                inside_l3d = (L3D_cm > win_min) and (L3D_cm < win_max)
                inside_r = (R_cm > win_min) and (R_cm < win_max)

                if inside_l3d:
                    n_truth_l3d_win += 1
                if inside_r:
                    n_truth_r_win += 1

                R_all.append(R_cm)
                L_all.append(L3D_cm)

                if has_reco:
                    R_reco.append(R_cm)
                    L_reco.append(L3D_cm)

                    if inside_l3d:
                        n_reco_inside_l3d += 1
                    else:
                        n_reco_outside_l3d += 1
                    if inside_r:
                        n_reco_inside_r += 1
                    else:
                        n_reco_outside_r += 1

        if save_events:
            event_rows.append({
                "file": str(root_path),
                "file_name": root_path.name,
                "event": iev,
                "has_alp": has_alp,
                "has_decay_vertex": has_decay_vertex,
                "alp_index": ia,
                "d1_index": id1,
                "d2_index": id2,
                "d1_pid": d1_pid,
                "d2_pid": d2_pid,
                "R_cm": R_cm,
                "L3D_cm": L3D_cm,
                "LLP_size": llp_size,
                "has_reco_llp": has_reco,
                "inside_L3D_window": inside_l3d,
                "inside_R_window": inside_r,
            })

    summary = {
        "file": str(root_path),
        "file_name": root_path.name,
        "ma_from_path_GeV": ma,
        "fa_from_path_GeV": fa,
        "particle_collection": particle_collection,
        "llp_size_branch": llp_size_branch or "",
        "llp_count_field": llp_count_field or "",
        "read_error": "",
        "N_total": n_total,
        "N_has_ALP": n_has_alp,
        "N_has_decay_vertex": n_has_decay_vertex,
        "N_LLP_size_gt_0": n_reco,
        "eff_reco_LLP_size_gt_0": n_reco / n_total if n_total else float("nan"),
        "N_truth_L3D_window": n_truth_l3d_win,
        "P_truth_L3D_window": n_truth_l3d_win / n_total if n_total else float("nan"),
        "N_truth_R_window": n_truth_r_win,
        "P_truth_R_window": n_truth_r_win / n_total if n_total else float("nan"),
        "N_reco_inside_L3D_window": n_reco_inside_l3d,
        "N_reco_outside_L3D_window": n_reco_outside_l3d,
        "N_reco_inside_R_window": n_reco_inside_r,
        "N_reco_outside_R_window": n_reco_outside_r,
        "R_all_min_cm": amin(R_all),
        "R_all_q01_cm": quant(R_all, 0.01),
        "R_all_q05_cm": quant(R_all, 0.05),
        "R_all_q50_cm": quant(R_all, 0.50),
        "R_all_q95_cm": quant(R_all, 0.95),
        "R_all_q99_cm": quant(R_all, 0.99),
        "R_all_max_cm": amax(R_all),
        "L3D_all_min_cm": amin(L_all),
        "L3D_all_q01_cm": quant(L_all, 0.01),
        "L3D_all_q05_cm": quant(L_all, 0.05),
        "L3D_all_q50_cm": quant(L_all, 0.50),
        "L3D_all_q95_cm": quant(L_all, 0.95),
        "L3D_all_q99_cm": quant(L_all, 0.99),
        "L3D_all_max_cm": amax(L_all),
        "R_reco_min_cm": amin(R_reco),
        "R_reco_q01_cm": quant(R_reco, 0.01),
        "R_reco_q05_cm": quant(R_reco, 0.05),
        "R_reco_q50_cm": quant(R_reco, 0.50),
        "R_reco_q95_cm": quant(R_reco, 0.95),
        "R_reco_q99_cm": quant(R_reco, 0.99),
        "R_reco_max_cm": amax(R_reco),
        "L3D_reco_min_cm": amin(L_reco),
        "L3D_reco_q01_cm": quant(L_reco, 0.01),
        "L3D_reco_q05_cm": quant(L_reco, 0.05),
        "L3D_reco_q50_cm": quant(L_reco, 0.50),
        "L3D_reco_q95_cm": quant(L_reco, 0.95),
        "L3D_reco_q99_cm": quant(L_reco, 0.99),
        "L3D_reco_max_cm": amax(L_reco),
    }
    return summary, event_rows


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    seen = set(fields)
    for row in rows[1:]:
        for k in row.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root_files = sorted(root_dir.rglob(args.pattern))
    if not root_files:
        raise RuntimeError(f"No ROOT files found under {root_dir} with pattern {args.pattern}")

    summaries = []
    all_events = []

    for i, root_path in enumerate(root_files, start=1):
        print(f"[{i}/{len(root_files)}] {root_path}")
        summary, events = process_file(
            root_path=root_path,
            tree_name=args.tree_name,
            alp_id=args.alp_id,
            max_events=args.max_events,
            win_min=args.llp_window_min_cm,
            win_max=args.llp_window_max_cm,
            save_events=args.save_events,
            print_branches=(args.print_branches > 0 and i <= args.print_branches),
        )
        summaries.append(summary)
        if args.save_events:
            all_events.extend(events)

    summary_path = out_dir / "delphes_llp_boundary_summary_by_file.csv"
    write_csv(summary_path, summaries)
    print(f"Saved summary: {summary_path}")

    if args.save_events:
        events_path = out_dir / "delphes_llp_boundary_events.csv"
        write_csv(events_path, all_events)
        print(f"Saved events: {events_path}")

    n_ok = sum(1 for r in summaries if not r.get("read_error"))
    n_fail = len(summaries) - n_ok
    print("")
    print(f"Files processed: {len(summaries)}")
    print(f"Files read successfully: {n_ok}")
    print(f"Files failed: {n_fail}")
    print("")
    print("Main columns to inspect:")
    print("  L3D_reco_q01_cm, L3D_reco_q99_cm")
    print("  R_reco_q01_cm, R_reco_q99_cm")
    print("  N_reco_inside_L3D_window, N_reco_outside_L3D_window")


if __name__ == "__main__":
    main()
