#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_csv_tau_visible_LLP_1_200cm_parallel.py

Purpose
-------
Scan Delphes ROOT files and keep the original generate_csv.py extraction logic,
while adding a tau-channel LLP geometry column.

For a -> tau+ tau- -> e/mu + neutrinos, the visible displaced object is the final
charged lepton from tau decay, not the tau production point. Therefore this script
uses the production vertex of final e/mu descendants of ALP tau daughters:

    L3D_tau_visible_cm = |r_visible_lepton - r_ALP_production| / 10

where Delphes Particle.X/Y/Z are assumed to be in mm, so division by 10 converts
to cm.

Main slim output columns:
    sample, ma, fa, width, total_LLPs, LLP_tau_1-200cm

Definitions
-----------
    total_LLPs          = sum(LLPs_size)
    LLP_tau_1-200cm    = sum(LLPs_size for events satisfying all of:
                            LLPs_size > 0,
                            ALP has tau daughters,
                            at least one final e/mu from tau decay exists,
                            1 cm < L3D_tau_visible_cm < 200 cm)

If each event has at most one reconstructed LLP candidate, LLP_tau_1-200cm is
identical to the number of selected events.
"""

from __future__ import annotations

import math
import os
import re
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import uproot


# -----------------------------------------------------------------------------
# User-level settings
# -----------------------------------------------------------------------------

BASE_DIR = "/home/wsy/MG5_aMC_v3_5_6_LLP/ALP_LLP"
TARGET_BRANCH = "LLPs_size"

OUTPUT_CSV = os.path.join(BASE_DIR, "alp_tau_llp_summary.csv")
OUTPUT_CSV_SLIM = os.path.join(BASE_DIR, "alp_tau_llp_summary_slim.csv")

ALP_ID = 9000001
TAU_ID = 15
VISIBLE_LEPTON_IDS = {11, 13}  # e, mu

# Strict LLP visible displacement window in cm: 1 cm < L3D_tau_visible_cm < 200 cm.
L3D_MIN_CM = 1.0
L3D_MAX_CM = 200.0

# Main added column requested for tau-channel visible LLP matching.
LLP_TAU_WINDOW_COLUMN = "LLP_tau_1-200cm"

# Only count events where ALP has tau daughters. Keep True for the a -> tau tau file.
REQUIRE_ALP_TO_TAU = True

# Protection against malformed decay records.
MAX_DECAY_STEPS = 200
MAX_DAUGHTER_SPAN = 50

# Number of parallel worker processes. For i5-12400 / 6C12T, 6--8 is usually reasonable.
N_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))


# -----------------------------------------------------------------------------
# File and tree utilities: kept close to original generate_csv.py
# -----------------------------------------------------------------------------

def find_root_files(base_dir: str) -> List[str]:
    """查找所有位于 */Events/run_01/*.root 的文件。"""
    root_files: List[str] = []
    for dirpath, _, filenames in os.walk(base_dir):
        if os.path.basename(dirpath) == "run_01" and os.path.basename(os.path.dirname(dirpath)) == "Events":
            for fname in filenames:
                if fname.endswith(".root"):
                    root_files.append(os.path.join(dirpath, fname))
    root_files.sort()
    return root_files


def find_tree_with_branch(root_file: str, branch_name: str) -> Optional[str]:
    """在 ROOT 文件中查找包含指定 branch 的 TTree。"""
    with uproot.open(root_file) as f:
        for key, classname in f.classnames(recursive=True).items():
            if "TTree" not in classname:
                continue
            try:
                tree = f[key]
                if branch_name in tree.keys():
                    return key
            except Exception:
                continue
    return None


def branch_suffix_match(branches: Sequence[str], collection: str, field: str) -> Optional[str]:
    """
    稳健匹配 Delphes 分支名。

    可匹配示例：
      Particle.PID
      Particle/Particle.PID
      Delphes/Particle.PID
      Particle/Particle.Particle.PID
    """
    exact_candidates = [
        f"{collection}.{field}",
        f"{collection}/{collection}.{field}",
        f"{collection}/{field}",
    ]

    bset = set(branches)
    for candidate in exact_candidates:
        if candidate in bset:
            return candidate

    suffixes = [
        f"{collection}.{field}",
        f"/{collection}.{field}",
        f"/{collection}/{collection}.{field}",
        f"/{collection}/{field}",
    ]

    matches: List[str] = []
    for branch in branches:
        branch_lower = branch.lower()
        for suffix in suffixes:
            if branch_lower.endswith(suffix.lower()):
                matches.append(branch)
                break

    if matches:
        return sorted(matches, key=len)[0]

    field_lower = field.lower()
    collection_lower = collection.lower()
    matches = [
        branch for branch in branches
        if collection_lower in branch.lower() and branch.lower().endswith("." + field_lower)
    ]
    if matches:
        return sorted(matches, key=len)[0]

    return None


def find_particle_branches(branches: Sequence[str]) -> Dict[str, Optional[str]]:
    """查找 tau-channel truth matching 所需的 Delphes Particle 分支。"""
    return {
        "PID": branch_suffix_match(branches, "Particle", "PID"),
        "D1": branch_suffix_match(branches, "Particle", "D1"),
        "D2": branch_suffix_match(branches, "Particle", "D2"),
        "X": branch_suffix_match(branches, "Particle", "X"),
        "Y": branch_suffix_match(branches, "Particle", "Y"),
        "Z": branch_suffix_match(branches, "Particle", "Z"),
    }


def to_event_array(x) -> np.ndarray:
    """将单个 event 的 jagged 内容统一转成 numpy array。"""
    return np.asarray(x)


def to_scalar_llp_size(x) -> int:
    """将单个 event 的 LLPs_size 转成整数。"""
    arr = np.asarray(x)
    try:
        if arr.shape == ():
            return int(arr.item())
        if len(arr) == 1:
            y = arr[0]
            return int(y.item() if hasattr(y, "item") else y)
        return int(arr.item())
    except Exception:
        try:
            return int(x)
        except Exception:
            return 0


def finite_min(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.min(arr))


def finite_max(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.max(arr))


def finite_quantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.quantile(arr, q))


# -----------------------------------------------------------------------------
# Decay-chain logic for a -> tau tau -> visible e/mu
# -----------------------------------------------------------------------------

def valid_index(index: int, n_particles: int) -> bool:
    return 0 <= index < n_particles


def as_int_safe(x, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def daughter_indices(d1_evt: np.ndarray, d2_evt: np.ndarray, parent_index: int, n_particles: int) -> List[int]:
    """
    Return daughter indices for a particle.

    Delphes Particle.D1/D2 are normally first/last daughter indices. For normal
    records we use the inclusive range [D1, D2]. If the range is suspiciously
    large or malformed, fall back to the valid endpoint indices only.
    """
    if not valid_index(parent_index, n_particles):
        return []

    d1 = as_int_safe(d1_evt[parent_index])
    d2 = as_int_safe(d2_evt[parent_index])

    candidates: List[int] = []

    if valid_index(d1, n_particles) and valid_index(d2, n_particles):
        lo, hi = min(d1, d2), max(d1, d2)
        span = hi - lo + 1
        if span <= MAX_DAUGHTER_SPAN:
            candidates = list(range(lo, hi + 1))
        else:
            candidates = [d1, d2]
    elif valid_index(d1, n_particles):
        candidates = [d1]
    elif valid_index(d2, n_particles):
        candidates = [d2]

    # Remove self-reference and duplicates while preserving order.
    out: List[int] = []
    seen: Set[int] = set()
    for idx in candidates:
        if idx == parent_index or not valid_index(idx, n_particles) or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def find_descendants_with_abs_pid(
    pid_evt: np.ndarray,
    d1_evt: np.ndarray,
    d2_evt: np.ndarray,
    start_index: int,
    target_abs_pids: Set[int],
    stop_at_match: bool,
) -> List[int]:
    """
    Breadth-first traversal of the Delphes truth decay chain.

    Parameters
    ----------
    start_index:
        Parent particle index. The parent itself is not tested; traversal starts
        from its daughters.
    target_abs_pids:
        Absolute PDG IDs to collect.
    stop_at_match:
        If True, do not continue below a matching particle.
    """
    n_particles = len(pid_evt)
    queue: deque[int] = deque(daughter_indices(d1_evt, d2_evt, start_index, n_particles))
    visited: Set[int] = {start_index}
    matches: List[int] = []
    steps = 0

    while queue and steps < MAX_DECAY_STEPS:
        idx = queue.popleft()
        if idx in visited or not valid_index(idx, n_particles):
            continue
        visited.add(idx)
        steps += 1

        abs_pid = abs(as_int_safe(pid_evt[idx]))
        if abs_pid in target_abs_pids:
            matches.append(idx)
            if stop_at_match:
                continue

        for child in daughter_indices(d1_evt, d2_evt, idx, n_particles):
            if child not in visited:
                queue.append(child)

    return matches


def find_alp_tau_daughters(
    pid_evt: np.ndarray,
    d1_evt: np.ndarray,
    d2_evt: np.ndarray,
    alp_index: int,
) -> List[int]:
    """
    Find tau daughters of the ALP.

    First try direct ALP daughters. If not found, fall back to a short decay-chain
    search, which helps if the event record contains intermediate copies.
    """
    n_particles = len(pid_evt)
    direct = daughter_indices(d1_evt, d2_evt, alp_index, n_particles)
    tau_direct = [idx for idx in direct if abs(as_int_safe(pid_evt[idx])) == TAU_ID]
    if tau_direct:
        return tau_direct

    # Fallback for event records with intermediate copies.
    tau_desc = find_descendants_with_abs_pid(
        pid_evt=pid_evt,
        d1_evt=d1_evt,
        d2_evt=d2_evt,
        start_index=alp_index,
        target_abs_pids={TAU_ID},
        stop_at_match=True,
    )
    return tau_desc


def find_visible_leptons_from_tau_chain(
    pid_evt: np.ndarray,
    d1_evt: np.ndarray,
    d2_evt: np.ndarray,
    tau_indices: Iterable[int],
) -> List[int]:
    """Find final visible e/mu descendants from tau decay chains."""
    visible: List[int] = []
    seen: Set[int] = set()

    for tau_idx in tau_indices:
        matches = find_descendants_with_abs_pid(
            pid_evt=pid_evt,
            d1_evt=d1_evt,
            d2_evt=d2_evt,
            start_index=tau_idx,
            target_abs_pids=VISIBLE_LEPTON_IDS,
            stop_at_match=True,
        )
        for idx in matches:
            if idx not in seen:
                seen.add(idx)
                visible.append(idx)

    return visible


def l3d_from_origin_cm(
    x_evt: np.ndarray,
    y_evt: np.ndarray,
    z_evt: np.ndarray,
    origin_index: int,
    visible_index: int,
) -> float:
    """
    Compute |r_visible - r_origin| in cm.

    Particle.X/Y/Z are assumed to be in mm, so divide by 10.
    For prompt ALP production, r_origin is the primary vertex and is usually near zero.
    """
    x0 = float(x_evt[origin_index])
    y0 = float(y_evt[origin_index])
    z0 = float(z_evt[origin_index])

    xv = float(x_evt[visible_index])
    yv = float(y_evt[visible_index])
    zv = float(z_evt[visible_index])

    dx = xv - x0
    dy = yv - y0
    dz = zv - z0
    return math.sqrt(dx * dx + dy * dy + dz * dz) / 10.0


# -----------------------------------------------------------------------------
# Per-file extraction
# -----------------------------------------------------------------------------

def get_total_llps_and_tau_visible_window(root_file: str, branch_name: str = "LLPs_size"):
    """
    Return per-file totals and tau-channel visible LLP matching.
    """
    tree_path = find_tree_with_branch(root_file, branch_name)
    if tree_path is None:
        raise RuntimeError(f"branch '{branch_name}' not found in {root_file}")

    with uproot.open(root_file) as f:
        tree = f[tree_path]
        branches = list(tree.keys())
        particle = find_particle_branches(branches)

        missing = [name for name, branch in particle.items() if branch is None]
        if missing:
            raise RuntimeError(
                f"missing Particle branches {missing} in {root_file}; "
                f"cannot construct tau visible L3D. Example branches: {branches[:30]}"
            )

        read_branches = [
            branch_name,
            particle["PID"],
            particle["D1"],
            particle["D2"],
            particle["X"],
            particle["Y"],
            particle["Z"],
        ]
        arrays = tree.arrays(read_branches, library="np")

    sizes_all = arrays[branch_name]
    pid_all = arrays[particle["PID"]]
    d1_all = arrays[particle["D1"]]
    d2_all = arrays[particle["D2"]]
    x_all = arrays[particle["X"]]
    y_all = arrays[particle["Y"]]
    z_all = arrays[particle["Z"]]

    total_events = int(len(sizes_all))
    total_llps = 0

    llp_tau_1_200cm = 0
    llp_tau_1_200cm_events = 0

    n_reco_events = 0
    n_reco_events_has_alp = 0
    n_reco_events_alp_to_tau = 0
    n_reco_events_has_tau_visible_lepton = 0

    n_tau_visible_leptons = 0
    n_tau_visible_leptons_in_window = 0

    visible_l3d_values_all_reco: List[float] = []
    visible_l3d_values_selected: List[float] = []

    for iev in range(total_events):
        llp_size = to_scalar_llp_size(sizes_all[iev])
        total_llps += llp_size

        if llp_size <= 0:
            continue
        n_reco_events += 1

        pid_evt = to_event_array(pid_all[iev])
        d1_evt = to_event_array(d1_all[iev])
        d2_evt = to_event_array(d2_all[iev])
        x_evt = to_event_array(x_all[iev])
        y_evt = to_event_array(y_all[iev])
        z_evt = to_event_array(z_all[iev])

        alp_indices = np.where(np.abs(pid_evt) == ALP_ID)[0]
        if len(alp_indices) == 0:
            continue
        n_reco_events_has_alp += 1

        event_inside_window = False
        event_has_tau = False
        event_has_visible = False

        # In normal samples there is one ALP, but allow more than one for robustness.
        for alp_index_raw in alp_indices:
            alp_index = int(alp_index_raw)
            tau_indices = find_alp_tau_daughters(pid_evt, d1_evt, d2_evt, alp_index)

            if REQUIRE_ALP_TO_TAU and not tau_indices:
                continue
            if tau_indices:
                event_has_tau = True

            visible_indices = find_visible_leptons_from_tau_chain(pid_evt, d1_evt, d2_evt, tau_indices)
            if visible_indices:
                event_has_visible = True

            for vis_index in visible_indices:
                l3d_cm = l3d_from_origin_cm(x_evt, y_evt, z_evt, alp_index, vis_index)
                visible_l3d_values_all_reco.append(l3d_cm)
                n_tau_visible_leptons += 1

                inside = (l3d_cm > L3D_MIN_CM) and (l3d_cm < L3D_MAX_CM)
                if inside:
                    event_inside_window = True
                    visible_l3d_values_selected.append(l3d_cm)
                    n_tau_visible_leptons_in_window += 1

        if event_has_tau:
            n_reco_events_alp_to_tau += 1
        if event_has_visible:
            n_reco_events_has_tau_visible_lepton += 1

        if event_inside_window:
            # Keep counting convention consistent with total_LLPs = sum(LLPs_size).
            llp_tau_1_200cm += llp_size
            llp_tau_1_200cm_events += 1

    diagnostics = {
        "n_reco_events_LLPs_size_gt_0": n_reco_events,
        "n_reco_events_has_ALP": n_reco_events_has_alp,
        "n_reco_events_ALP_to_tau": n_reco_events_alp_to_tau,
        "n_reco_events_has_tau_visible_e_mu": n_reco_events_has_tau_visible_lepton,
        "LLP_tau_1-200cm_events": llp_tau_1_200cm_events,
        "eff_tau_1_200cm_events": llp_tau_1_200cm_events / total_events if total_events else float("nan"),
        "n_tau_visible_e_mu": n_tau_visible_leptons,
        "n_tau_visible_e_mu_in_window": n_tau_visible_leptons_in_window,
        "L3D_tau_visible_min_cm": finite_min(visible_l3d_values_all_reco),
        "L3D_tau_visible_q01_cm": finite_quantile(visible_l3d_values_all_reco, 0.01),
        "L3D_tau_visible_q50_cm": finite_quantile(visible_l3d_values_all_reco, 0.50),
        "L3D_tau_visible_q99_cm": finite_quantile(visible_l3d_values_all_reco, 0.99),
        "L3D_tau_visible_max_cm": finite_max(visible_l3d_values_all_reco),
        "L3D_tau_visible_selected_min_cm": finite_min(visible_l3d_values_selected),
        "L3D_tau_visible_selected_max_cm": finite_max(visible_l3d_values_selected),
        "particle_pid_branch": particle["PID"],
        "particle_d1_branch": particle["D1"],
        "particle_d2_branch": particle["D2"],
        "particle_x_branch": particle["X"],
        "particle_y_branch": particle["Y"],
        "particle_z_branch": particle["Z"],
    }

    return total_llps, total_events, llp_tau_1_200cm, tree_path, diagnostics


def parse_banner_file(banner_file: str) -> Tuple[float, float, float]:
    """
    从 run_01_tag_1_banner.txt 中提取：
      - ma    : 9000001 10 # ma 这一行中的第二列
      - fa    : 1 1000000 # fa 这一行中的第二列
      - width : DECAY 9000001 1.700000e-16 这一行中的第三列
    """
    if not os.path.exists(banner_file):
        raise FileNotFoundError(f"banner file not found: {banner_file}")

    ma = None
    fa = None
    width = None

    ma_pattern = re.compile(r"^\s*9000001\s+([^\s#]+)\s*#\s*ma\s*$", re.IGNORECASE)
    fa_pattern = re.compile(r"^\s*1\s+([^\s#]+)\s*#\s*fa\s*$", re.IGNORECASE)
    width_pattern = re.compile(r"^\s*DECAY\s+9000001\s+([^\s#]+)", re.IGNORECASE)

    with open(banner_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_strip = line.strip()

            if ma is None:
                m = ma_pattern.match(line_strip)
                if m:
                    ma = float(m.group(1))
                    continue

            if fa is None:
                m = fa_pattern.match(line_strip)
                if m:
                    fa = float(m.group(1))
                    continue

            if width is None:
                m = width_pattern.match(line_strip)
                if m:
                    width = float(m.group(1))
                    continue

    if ma is None:
        raise ValueError(f"cannot find ma in {banner_file}")
    if fa is None:
        raise ValueError(f"cannot find fa in {banner_file}")
    if width is None:
        raise ValueError(f"cannot find width in {banner_file}")

    return ma, fa, width


def process_one_root(root_file: str, base_dir: str) -> Dict[str, object]:
    """对单个 root 文件提取 sample, ma, fa, width, total_LLPs, LLP_tau_1-200cm。"""
    rel_path = os.path.relpath(root_file, base_dir)
    parts = rel_path.split(os.sep)
    sample = parts[0] if len(parts) > 0 else ""

    run_dir = os.path.dirname(root_file)
    banner_file = os.path.join(run_dir, "run_01_tag_1_banner.txt")

    ma, fa, width = parse_banner_file(banner_file)
    total_llps, total_events, llp_tau_1_200cm, tree_path, diagnostics = get_total_llps_and_tau_visible_window(
        root_file,
        TARGET_BRANCH,
    )

    result: Dict[str, object] = {
        "sample": sample,
        "ma": ma,
        "fa": fa,
        "width": width,
        "total_LLPs": total_llps,
        LLP_TAU_WINDOW_COLUMN: llp_tau_1_200cm,
        "total_events": total_events,
        "root_file": root_file,
        "banner_file": banner_file,
        "tree": tree_path,
        "status": "ok",
    }
    result.update(diagnostics)
    return result


def make_error_result(root_file: str, error) -> Dict[str, object]:
    return {
        "sample": "",
        "ma": np.nan,
        "fa": np.nan,
        "width": np.nan,
        "total_LLPs": np.nan,
        LLP_TAU_WINDOW_COLUMN: np.nan,
        "total_events": np.nan,
        "root_file": root_file,
        "banner_file": os.path.join(os.path.dirname(root_file), "run_01_tag_1_banner.txt"),
        "tree": "",
        "status": f"error: {error}",
    }


def process_one_root_safe(task):
    """Multiprocessing wrapper preserving deterministic order."""
    order, root_file, base_dir = task
    try:
        result = process_one_root(root_file, base_dir)
    except Exception as e:
        result = make_error_result(root_file, e)
    result["_order"] = order
    return order, result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    root_files = find_root_files(BASE_DIR)

    if not root_files:
        print("没有找到任何位于 */Events/run_01/*.root 的文件")
        return

    n_workers = max(1, int(N_WORKERS))
    n_workers = min(n_workers, len(root_files))

    print(f"共找到 {len(root_files)} 个 root 文件\n")
    print(f"tau visible L3D cut: {L3D_MIN_CM} cm < L3D_tau_visible_cm < {L3D_MAX_CM} cm")
    print(f"ALP_ID = {ALP_ID}, TAU_ID = {TAU_ID}, VISIBLE_LEPTON_IDS = {sorted(VISIBLE_LEPTON_IDS)}")
    print(f"新增输出列: {LLP_TAU_WINDOW_COLUMN}")
    print(f"并行进程数: {n_workers}\n")

    tasks = [(i, root_file, BASE_DIR) for i, root_file in enumerate(root_files)]
    results: List[Dict[str, object]] = []
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_one_root_safe, task): task[1] for task in tasks}

        for future in as_completed(futures):
            root_file = futures[future]
            done += 1
            try:
                _, result = future.result()
            except Exception as e:
                result = make_error_result(root_file, e)
                result["_order"] = root_files.index(root_file)

            results.append(result)

            if done == 1 or done % 50 == 0 or done == len(root_files):
                print(f"Processed {done}/{len(root_files)} files")

    results.sort(key=lambda row: row.get("_order", 10**18))
    for row in results:
        row.pop("_order", None)

    df = pd.DataFrame(results)

    # 主表：保留原来的列，并新增 tau 通道 LLP 几何筛选列。
    main_cols = ["sample", "ma", "fa", "width", "total_LLPs", LLP_TAU_WINDOW_COLUMN]
    df_main = df[main_cols].copy()

    df.to_csv(OUTPUT_CSV, index=False)
    df_main.to_csv(OUTPUT_CSV_SLIM, index=False)

    print("\n已保存：")
    print(OUTPUT_CSV)
    print(OUTPUT_CSV_SLIM)

    ok_df = df[df["status"] == "ok"]
    if not ok_df.empty:
        print("\n前几行结果预览：")
        print(ok_df[main_cols].head().to_string(index=False))

    failed_df = df[df["status"] != "ok"]
    if not failed_df.empty:
        print("\n以下文件处理失败：")
        print(failed_df[["root_file", "status"]].to_string(index=False))

    print("\n诊断列建议查看：")
    print("  n_reco_events_ALP_to_tau")
    print("  n_reco_events_has_tau_visible_e_mu")
    print("  LLP_tau_1-200cm_events")
    print("  eff_tau_1_200cm_events")
    print("  L3D_tau_visible_q01_cm, L3D_tau_visible_q99_cm")


if __name__ == "__main__":
    main()
