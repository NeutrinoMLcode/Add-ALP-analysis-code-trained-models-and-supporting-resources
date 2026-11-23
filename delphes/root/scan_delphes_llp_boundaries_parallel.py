#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_delphes_llp_boundaries_parallel.py

Parallel scanner for Delphes ROOT files.

Purpose
-------
Recursively scan many Delphes ROOT files and estimate the effective displacement
range of events with LLP_size > 0.

This script avoids:
  - awkward
  - pandas
  - pyarrow

Dependencies:
  - uproot
  - numpy

Main output:
  delphes_llp_boundary_summary_by_file.csv

Optional output:
  delphes_llp_boundary_events.csv  (only if --save-events is used)

Recommended first run:
  python scan_delphes_llp_boundaries_parallel.py \
    --root-dir "/home/wsy/MG5_aMC_v3_5_6_LLP/ALP_LLP" \
    --out-dir "/home/wsy/MG5_aMC_v3_5_6_LLP/ALP_LLP/llp_boundary_check_parallel" \
    --alp-id 9000001 \
    --pattern "tag_1_delphes_events.root" \
    --n-workers 8 \
    --print-branches 1

If ROOT file names are not uniform:
  --pattern "*.root"

Notes
-----
- R_cm and L3D_cm are computed from the first daughter vertex of ALP.
- Delphes Particle.X/Y/Z are assumed to be in mm, so this script divides by 10.
- reco columns mean events with LLP_size > 0.
- q01--q99 gives a rough effective reconstructed displacement range.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import uproot


SUMMARY_COLUMNS = [
    "file",
    "file_name",
    "ma_from_path_GeV",
    "fa_from_path_GeV",
    "particle_pid_branch",
    "particle_d1_branch",
    "particle_d2_branch",
    "particle_x_branch",
    "particle_y_branch",
    "particle_z_branch",
    "llp_size_branch",
    "llp_count_mode",
    "read_error",

    "N_total",
    "N_has_ALP",
    "N_has_decay_vertex",
    "N_LLP_size_gt_0",
    "eff_reco_LLP_size_gt_0",

    "N_truth_L3D_window",
    "P_truth_L3D_window",
    "N_truth_R_window",
    "P_truth_R_window",
    "N_reco_inside_L3D_window",
    "N_reco_outside_L3D_window",
    "N_reco_inside_R_window",
    "N_reco_outside_R_window",

    "R_all_min_cm",
    "R_all_q01_cm",
    "R_all_q05_cm",
    "R_all_q50_cm",
    "R_all_q95_cm",
    "R_all_q99_cm",
    "R_all_max_cm",

    "L3D_all_min_cm",
    "L3D_all_q01_cm",
    "L3D_all_q05_cm",
    "L3D_all_q50_cm",
    "L3D_all_q95_cm",
    "L3D_all_q99_cm",
    "L3D_all_max_cm",

    "R_reco_min_cm",
    "R_reco_q01_cm",
    "R_reco_q05_cm",
    "R_reco_q50_cm",
    "R_reco_q95_cm",
    "R_reco_q99_cm",
    "R_reco_max_cm",

    "L3D_reco_min_cm",
    "L3D_reco_q01_cm",
    "L3D_reco_q05_cm",
    "L3D_reco_q50_cm",
    "L3D_reco_q95_cm",
    "L3D_reco_q99_cm",
    "L3D_reco_max_cm",
]

EVENT_COLUMNS = [
    "file",
    "file_name",
    "event",
    "has_alp",
    "has_decay_vertex",
    "alp_index",
    "d1_index",
    "d2_index",
    "d1_pid",
    "d2_pid",
    "R_cm",
    "L3D_cm",
    "LLP_size",
    "has_reco_llp",
    "inside_L3D_window",
    "inside_R_window",
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--root-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tree-name", default="Delphes")
    p.add_argument("--alp-id", type=int, default=9000001)
    p.add_argument("--pattern", default="*.root")
    p.add_argument("--max-events", type=int, default=None)

    p.add_argument("--llp-window-min-cm", type=float, default=1.0)
    p.add_argument("--llp-window-max-cm", type=float, default=200.0)

    p.add_argument(
        "--n-workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="Number of parallel processes. Default: min(8, ncpu-1).",
    )

    p.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Task chunk size for process pool. Keep 1 for uneven ROOT file sizes.",
    )

    p.add_argument(
        "--save-events",
        action="store_true",
        help="Save event-level CSV. This can be very large and slows down the run.",
    )

    p.add_argument(
        "--print-branches",
        type=int,
        default=0,
        help="Print branch names for the first N ROOT files and exit after normal scan starts. Use 1 for debugging.",
    )

    return p.parse_args()


def parse_ma_fa_from_path(path: Path) -> Tuple[float, float]:
    text = str(path)

    # Common path style: alp_ma_0p4GeV_fa_1TeV
    m_ma = re.search(r"ma[_-]?(?P<ma>\d+(?:\.\d+)?|\d+p\d+)\s*gev", text, re.I)
    m_fa = re.search(r"(fa|lambda)[_-]?(?P<fa>\d+(?:\.\d+)?|\d+p\d+)\s*(?P<unit>gev|tev)", text, re.I)

    if m_ma and m_fa:
        ma = float(m_ma.group("ma").replace("p", "."))
        fa = float(m_fa.group("fa").replace("p", "."))
        if m_fa.group("unit").lower() == "tev":
            fa *= 1000.0
        return ma, fa

    # Common file style: 1.4geV-4000GeV
    m = re.search(
        r"(?P<ma>\d+(?:\.\d+)?|\d+p\d+)\s*gev\s*[-_]\s*(?P<fa>\d+(?:\.\d+)?|\d+p\d+)\s*(?P<unit>gev|tev)",
        text,
        re.I,
    )

    if m:
        ma = float(m.group("ma").replace("p", "."))
        fa = float(m.group("fa").replace("p", "."))
        if m.group("unit").lower() == "tev":
            fa *= 1000.0
        return ma, fa

    return float("nan"), float("nan")


def finite_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def quantile(values: Sequence[float], prob: float) -> float:
    arr = finite_array(values)
    if len(arr) == 0:
        return float("nan")
    return float(np.quantile(arr, prob))


def arr_min(values: Sequence[float]) -> float:
    arr = finite_array(values)
    if len(arr) == 0:
        return float("nan")
    return float(np.min(arr))


def arr_max(values: Sequence[float]) -> float:
    arr = finite_array(values)
    if len(arr) == 0:
        return float("nan")
    return float(np.max(arr))


def branch_suffix_match(branches: Sequence[str], collection: str, field: str) -> Optional[str]:
    """
    Match Delphes branches robustly.

    Examples this can match:
      Particle.PID
      Particle/Particle.PID
      Delphes/Particle.PID
      Particle/Particle.Particle.PID  (rare)
    """
    exact_candidates = [
        f"{collection}.{field}",
        f"{collection}/{collection}.{field}",
        f"{collection}/{field}",
    ]

    bset = set(branches)
    for c in exact_candidates:
        if c in bset:
            return c

    suffixes = [
        f"{collection}.{field}",
        f"/{collection}.{field}",
        f"/{collection}/{collection}.{field}",
        f"/{collection}/{field}",
    ]

    matches = []
    for b in branches:
        bl = b.lower()
        for s in suffixes:
            if bl.endswith(s.lower()):
                matches.append(b)
                break

    if matches:
        return sorted(matches, key=len)[0]

    # Fallback: require both collection and field tokens.
    field_l = field.lower()
    collection_l = collection.lower()
    matches = [
        b for b in branches
        if collection_l in b.lower() and b.lower().endswith("." + field_l)
    ]
    if matches:
        return sorted(matches, key=len)[0]

    return None


def find_particle_branches(branches: Sequence[str]) -> Dict[str, Optional[str]]:
    # Default Delphes collection name is Particle.
    out = {
        "PID": branch_suffix_match(branches, "Particle", "PID"),
        "D1": branch_suffix_match(branches, "Particle", "D1"),
        "D2": branch_suffix_match(branches, "Particle", "D2"),
        "X": branch_suffix_match(branches, "Particle", "X"),
        "Y": branch_suffix_match(branches, "Particle", "Y"),
        "Z": branch_suffix_match(branches, "Particle", "Z"),
    }
    return out


def find_llp_size_branch(branches: Sequence[str]) -> Tuple[Optional[str], str]:
    """
    Return:
      branch name, mode

    mode:
      explicit_size: use branch scalar directly.
      candidate_array: infer size from a jagged candidate field.
      missing: no LLP branch found.
    """
    bset = set(branches)

    explicit = [
        "LLP_size",
        "LLPs_size",
        "LLP/LLP_size",
        "LLPs/LLPs_size",
    ]

    for b in explicit:
        if b in bset:
            return b, "explicit_size"

    size_matches = [
        b for b in branches
        if ("llp" in b.lower()) and b.lower().endswith("_size")
    ]

    if size_matches:
        return sorted(size_matches, key=len)[0], "explicit_size"

    # If no *_size exists, try any LLP candidate field as jagged array.
    # Common TreeWriter branch from card: LLPs LLPCandidate.
    candidate_preferences = [
        "LLPs.PT",
        "LLPs/LLPs.PT",
        "LLP.PT",
        "LLP/LLP.PT",
        "LLPs.Mass",
        "LLPs/LLPs.Mass",
        "LLPs.Eta",
        "LLPs/LLPs.Eta",
    ]

    for b in candidate_preferences:
        if b in bset:
            return b, "candidate_array"

    candidates = [
        b for b in branches
        if "llp" in b.lower() and (
            b.lower().endswith(".pt")
            or b.lower().endswith(".eta")
            or b.lower().endswith(".phi")
            or b.lower().endswith(".mass")
            or b.lower().endswith(".e")
        )
    ]

    if candidates:
        return sorted(candidates, key=len)[0], "candidate_array"

    return None, "missing"


def to_event_array(x) -> np.ndarray:
    return np.asarray(x)


def to_scalar_or_len(x, mode: str):
    arr = np.asarray(x)

    if mode == "candidate_array":
        try:
            return len(arr)
        except Exception:
            return 0

    try:
        if arr.shape == ():
            return arr.item()
        if len(arr) == 1:
            y = arr[0]
            return y.item() if hasattr(y, "item") else y
        return arr.item()
    except Exception:
        try:
            return float(x)
        except Exception:
            return float("nan")


def make_empty_summary(root_path: Path, error_msg: str) -> Dict[str, object]:
    ma, fa = parse_ma_fa_from_path(root_path)

    row = {k: float("nan") for k in SUMMARY_COLUMNS}

    row.update({
        "file": str(root_path),
        "file_name": root_path.name,
        "ma_from_path_GeV": ma,
        "fa_from_path_GeV": fa,
        "particle_pid_branch": "",
        "particle_d1_branch": "",
        "particle_d2_branch": "",
        "particle_x_branch": "",
        "particle_y_branch": "",
        "particle_z_branch": "",
        "llp_size_branch": "",
        "llp_count_mode": "",
        "read_error": error_msg,
    })

    return row


def inspect_branches(root_path: str, tree_name: str, max_print: int = 250) -> List[str]:
    root_path = str(root_path)
    out = []
    try:
        with uproot.open(root_path) as f:
            if tree_name not in f:
                out.append(f"[{root_path}] tree {tree_name!r} not found. keys={list(f.keys())[:20]}")
                return out

            tree = f[tree_name]
            branches = list(tree.keys())
            out.append(f"\n[{root_path}]")
            out.append(f"N branches = {len(branches)}")
            for b in branches[:max_print]:
                out.append(str(b))
            if len(branches) > max_print:
                out.append(f"... truncated, {len(branches) - max_print} more branches")
    except Exception as e:
        out.append(f"[{root_path}] branch inspection failed: {repr(e)}")
    return out


def process_one_file(task: Tuple[str, str, int, Optional[int], float, float, bool]) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    root_path_s, tree_name, alp_id, max_events, win_min, win_max, save_events = task
    root_path = Path(root_path_s)
    ma, fa = parse_ma_fa_from_path(root_path)

    try:
        with uproot.open(root_path) as f:
            if tree_name not in f:
                return make_empty_summary(root_path, f"Tree {tree_name!r} not found. keys={list(f.keys())[:20]}"), []

            tree = f[tree_name]
            branches = list(tree.keys())

            particle = find_particle_branches(branches)
            missing = [k for k, v in particle.items() if v is None]
            if missing:
                return make_empty_summary(
                    root_path,
                    f"Missing Particle branches {missing}. Example branches: {branches[:30]}",
                ), []

            llp_branch, llp_mode = find_llp_size_branch(branches)

            read_branches = [
                particle["PID"],
                particle["D1"],
                particle["D2"],
                particle["X"],
                particle["Y"],
                particle["Z"],
            ]

            if llp_branch is not None:
                read_branches.append(llp_branch)

            arrays = tree.arrays(read_branches, library="np", entry_stop=max_events)

    except Exception as e:
        return make_empty_summary(root_path, repr(e) + "\n" + traceback.format_exc(limit=3)), []

    pid_all = arrays[particle["PID"]]
    d1_all = arrays[particle["D1"]]
    d2_all = arrays[particle["D2"]]
    x_all = arrays[particle["X"]]
    y_all = arrays[particle["Y"]]
    z_all = arrays[particle["Z"]]

    llp_all = arrays[llp_branch] if llp_branch is not None else None

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

    R_all = []
    L_all = []
    R_reco = []
    L_reco = []
    event_rows = []

    for iev in range(n_total):
        pid_evt = to_event_array(pid_all[iev])
        d1_evt = to_event_array(d1_all[iev])
        d2_evt = to_event_array(d2_all[iev])
        x_evt = to_event_array(x_all[iev])
        y_evt = to_event_array(y_all[iev])
        z_evt = to_event_array(z_all[iev])

        if llp_all is None:
            llp_size = 0
            has_reco = False
        else:
            llp_size = to_scalar_or_len(llp_all[iev], llp_mode)
            try:
                has_reco = bool(llp_size > 0)
            except Exception:
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
        ia_out = ""
        id1_out = ""
        id2_out = ""
        inside_l3d = False
        inside_r = False

        if has_alp:
            n_has_alp += 1
            ia = int(alp_indices[0])
            ia_out = ia

            id1 = int(d1_evt[ia])
            id2 = int(d2_evt[ia])
            id1_out = id1
            id2_out = id2

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
                "alp_index": ia_out,
                "d1_index": id1_out,
                "d2_index": id2_out,
                "d1_pid": d1_pid,
                "d2_pid": d2_pid,
                "R_cm": R_cm,
                "L3D_cm": L3D_cm,
                "LLP_size": llp_size,
                "has_reco_llp": has_reco,
                "inside_L3D_window": inside_l3d,
                "inside_R_window": inside_r,
            })

    summary = {k: float("nan") for k in SUMMARY_COLUMNS}

    summary.update({
        "file": str(root_path),
        "file_name": root_path.name,
        "ma_from_path_GeV": ma,
        "fa_from_path_GeV": fa,

        "particle_pid_branch": particle["PID"],
        "particle_d1_branch": particle["D1"],
        "particle_d2_branch": particle["D2"],
        "particle_x_branch": particle["X"],
        "particle_y_branch": particle["Y"],
        "particle_z_branch": particle["Z"],
        "llp_size_branch": llp_branch or "",
        "llp_count_mode": llp_mode,
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

        "R_all_min_cm": arr_min(R_all),
        "R_all_q01_cm": quantile(R_all, 0.01),
        "R_all_q05_cm": quantile(R_all, 0.05),
        "R_all_q50_cm": quantile(R_all, 0.50),
        "R_all_q95_cm": quantile(R_all, 0.95),
        "R_all_q99_cm": quantile(R_all, 0.99),
        "R_all_max_cm": arr_max(R_all),

        "L3D_all_min_cm": arr_min(L_all),
        "L3D_all_q01_cm": quantile(L_all, 0.01),
        "L3D_all_q05_cm": quantile(L_all, 0.05),
        "L3D_all_q50_cm": quantile(L_all, 0.50),
        "L3D_all_q95_cm": quantile(L_all, 0.95),
        "L3D_all_q99_cm": quantile(L_all, 0.99),
        "L3D_all_max_cm": arr_max(L_all),

        "R_reco_min_cm": arr_min(R_reco),
        "R_reco_q01_cm": quantile(R_reco, 0.01),
        "R_reco_q05_cm": quantile(R_reco, 0.05),
        "R_reco_q50_cm": quantile(R_reco, 0.50),
        "R_reco_q95_cm": quantile(R_reco, 0.95),
        "R_reco_q99_cm": quantile(R_reco, 0.99),
        "R_reco_max_cm": arr_max(R_reco),

        "L3D_reco_min_cm": arr_min(L_reco),
        "L3D_reco_q01_cm": quantile(L_reco, 0.01),
        "L3D_reco_q05_cm": quantile(L_reco, 0.05),
        "L3D_reco_q50_cm": quantile(L_reco, 0.50),
        "L3D_reco_q95_cm": quantile(L_reco, 0.95),
        "L3D_reco_q99_cm": quantile(L_reco, 0.99),
        "L3D_reco_max_cm": arr_max(L_reco),
    })

    return summary, event_rows


def write_csv(path: Path, rows: List[Dict[str, object]], columns: Optional[List[str]] = None):
    if columns is None:
        if not rows:
            columns = []
        else:
            columns = list(rows[0].keys())
            seen = set(columns)
            for row in rows[1:]:
                for k in row.keys():
                    if k not in seen:
                        columns.append(k)
                        seen.add(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        if not columns:
            return
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
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

    print(f"Found {len(root_files)} ROOT files")
    print(f"Using n_workers = {args.n_workers}")
    print(f"Output dir = {out_dir}")

    if args.print_branches > 0:
        n = min(args.print_branches, len(root_files))
        print(f"\nPrinting branches for first {n} file(s):")
        for i in range(n):
            for line in inspect_branches(str(root_files[i]), args.tree_name):
                print(line)
        print("")

    tasks = [
        (
            str(p),
            args.tree_name,
            args.alp_id,
            args.max_events,
            args.llp_window_min_cm,
            args.llp_window_max_cm,
            args.save_events,
        )
        for p in root_files
    ]

    summaries: List[Dict[str, object]] = []
    all_events: List[Dict[str, object]] = []

    done = 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = [ex.submit(process_one_file, task) for task in tasks]

        for fut in as_completed(futures):
            done += 1

            try:
                summary, events = fut.result()
            except Exception as e:
                # This should be rare because process_one_file catches most errors.
                summary = make_empty_summary(Path("UNKNOWN"), repr(e) + "\n" + traceback.format_exc(limit=3))
                events = []

            summaries.append(summary)

            if args.save_events and events:
                all_events.extend(events)

            if done == 1 or done % 50 == 0 or done == len(futures):
                print(f"Processed {done}/{len(futures)} files")

    summaries = sorted(summaries, key=lambda r: str(r.get("file", "")))

    summary_path = out_dir / "delphes_llp_boundary_summary_by_file.csv"
    write_csv(summary_path, summaries, SUMMARY_COLUMNS)
    print(f"Saved summary: {summary_path}")

    if args.save_events:
        events_path = out_dir / "delphes_llp_boundary_events.csv"
        write_csv(events_path, all_events, EVENT_COLUMNS)
        print(f"Saved events: {events_path}")

    n_error = sum(1 for r in summaries if str(r.get("read_error", "")) not in ("", "nan"))
    n_ok = len(summaries) - n_error
    print("")
    print(f"Files read successfully: {n_ok}")
    print(f"Files with read_error:  {n_error}")
    print("")
    print("Main columns to inspect:")
    print("  L3D_reco_q01_cm, L3D_reco_q99_cm")
    print("  R_reco_q01_cm, R_reco_q99_cm")
    print("  N_reco_inside_L3D_window, N_reco_outside_L3D_window")
    print("")
    print("Interpretation:")
    print("  reco means events with LLP_size > 0.")
    print("  q01--q99 gives a rough effective displacement range.")
    print("  If reco q01 < 1 cm, Delphes-LLP is reconstructing candidates below your 1 cm cut.")


if __name__ == "__main__":
    main()
