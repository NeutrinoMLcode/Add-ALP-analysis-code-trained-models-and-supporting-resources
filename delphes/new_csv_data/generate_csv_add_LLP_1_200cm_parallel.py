#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import uproot

BASE_DIR = "/home/wsy/MG5_aMC_v3_5_6_LLP/ALP_LLP"
TARGET_BRANCH = "LLPs_size"
OUTPUT_CSV = os.path.join(BASE_DIR, "alp_llp_summary.csv")

# ALP/LLP PDG ID used to find the truth particle in Delphes Particle collection.
ALP_ID = 9000001

# Strict LLP decay-length window in cm: 1 cm < L3D_cm < 200 cm.
L3D_MIN_CM = 1.0
L3D_MAX_CM = 200.0

# CSV column name requested by the user.
LLP_WINDOW_COLUMN = "LLP_1-200cm"

# Number of parallel worker processes used to process independent ROOT files.
# For i5-12400 / 6C12T, 6--8 is usually reasonable.
N_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))


def find_root_files(base_dir):
    """
    查找所有位于 */Events/run_01/*.root 的文件。
    保持原 generate_csv.py 的 root 文件搜索方式。
    """
    root_files = []
    for dirpath, _, filenames in os.walk(base_dir):
        if os.path.basename(dirpath) == "run_01" and os.path.basename(os.path.dirname(dirpath)) == "Events":
            for fname in filenames:
                if fname.endswith(".root"):
                    root_files.append(os.path.join(dirpath, fname))
    root_files.sort()
    return root_files


def find_tree_with_branch(root_file, branch_name):
    """
    在 root 文件中查找包含指定 branch 的 TTree。
    返回 tree_path，找不到则返回 None。
    保持原 generate_csv.py 的 tree 定位逻辑。
    """
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

    matches = []
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
    """
    查找构造 L3D_cm 所需的 Delphes Particle 分支。
    """
    return {
        "PID": branch_suffix_match(branches, "Particle", "PID"),
        "D1": branch_suffix_match(branches, "Particle", "D1"),
        "D2": branch_suffix_match(branches, "Particle", "D2"),
        "X": branch_suffix_match(branches, "Particle", "X"),
        "Y": branch_suffix_match(branches, "Particle", "Y"),
        "Z": branch_suffix_match(branches, "Particle", "Z"),
    }


def to_event_array(x) -> np.ndarray:
    """
    将单个 event 的 jagged 内容统一转成 numpy array。
    """
    return np.asarray(x)


def to_scalar_llp_size(x) -> int:
    """
    将单个 event 的 LLPs_size 转成整数。
    TARGET_BRANCH 是显式 size 分支，因此这里按 scalar size 处理。
    """
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


def finite_min(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.min(arr))


def finite_max(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.max(arr))


def get_total_llps_and_l3d_window(root_file, branch_name="LLPs_size"):
    """
    读取 root 文件中的 LLPs_size，并返回：
      - total_LLPs = sum(LLPs_size)
      - total_events = len(LLPs_size)
      - LLP_1-200cm = sum(LLPs_size for events with LLPs_size > 0 and 1 cm < L3D_cm < 200 cm)

    L3D_cm 的构造方式参考 scan_delphes_llp_boundaries_parallel.py：
      1. 找到 Particle.PID 中 abs(PID) == ALP_ID 的 ALP；
      2. 读取该 ALP 的第一个 daughter 索引 Particle.D1；
      3. 使用第一个 daughter 的 Particle.X/Y/Z 作为衰变顶点；
      4. L3D_cm = sqrt(X_d^2 + Y_d^2 + Z_d^2) / 10.0。

    这里 /10.0 是将 Delphes Particle.X/Y/Z 常见的 mm 单位换成 cm。
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
                f"cannot construct L3D_cm. Example branches: {branches[:30]}"
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

    total_llps = int(np.sum(sizes_all, dtype=np.int64))
    total_events = int(len(sizes_all))

    llp_1_200cm = 0
    llp_1_200cm_events = 0
    n_reco_events = 0
    n_has_alp = 0
    n_has_decay_vertex = 0
    l3d_reco_values = []

    for iev in range(total_events):
        llp_size = to_scalar_llp_size(sizes_all[iev])
        has_reco_llp = llp_size > 0
        if not has_reco_llp:
            continue

        n_reco_events += 1

        pid_evt = to_event_array(pid_all[iev])
        d1_evt = to_event_array(d1_all[iev])
        d2_evt = to_event_array(d2_all[iev])  # 保留读取，便于和参考脚本的分支需求一致。
        x_evt = to_event_array(x_all[iev])
        y_evt = to_event_array(y_all[iev])
        z_evt = to_event_array(z_all[iev])

        alp_indices = np.where(np.abs(pid_evt) == ALP_ID)[0]
        if len(alp_indices) == 0:
            continue
        n_has_alp += 1

        alp_index = int(alp_indices[0])
        daughter1_index = int(d1_evt[alp_index])

        if not (0 <= daughter1_index < len(pid_evt)):
            continue
        n_has_decay_vertex += 1

        xd = float(x_evt[daughter1_index])
        yd = float(y_evt[daughter1_index])
        zd = float(z_evt[daughter1_index])

        # 与参考脚本一致：使用第一个 daughter 的顶点坐标构造 L3D_cm。
        l3d_cm = math.sqrt(xd * xd + yd * yd + zd * zd) / 10.0
        l3d_reco_values.append(l3d_cm)

        inside_l3d_window = (l3d_cm > L3D_MIN_CM) and (l3d_cm < L3D_MAX_CM)
        if inside_l3d_window:
            # 为了和 total_LLPs = sum(LLPs_size) 的计数口径一致，这里累加 LLPs_size。
            # 如果每个 event 最多一个 LLP，则该数值等同于通过 cut 的 event 数。
            llp_1_200cm += llp_size
            llp_1_200cm_events += 1

    diagnostics = {
        "n_reco_events_LLPs_size_gt_0": n_reco_events,
        "n_reco_events_has_ALP": n_has_alp,
        "n_reco_events_has_decay_vertex": n_has_decay_vertex,
        "LLP_1-200cm_events": llp_1_200cm_events,
        "L3D_reco_min_cm": finite_min(l3d_reco_values),
        "L3D_reco_max_cm": finite_max(l3d_reco_values),
        "particle_pid_branch": particle["PID"],
        "particle_d1_branch": particle["D1"],
        "particle_d2_branch": particle["D2"],
        "particle_x_branch": particle["X"],
        "particle_y_branch": particle["Y"],
        "particle_z_branch": particle["Z"],
    }

    return total_llps, total_events, llp_1_200cm, tree_path, diagnostics


def parse_banner_file(banner_file):
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

    ma_pattern = re.compile(r'^\s*9000001\s+([^\s#]+)\s*#\s*ma\s*$', re.IGNORECASE)
    fa_pattern = re.compile(r'^\s*1\s+([^\s#]+)\s*#\s*fa\s*$', re.IGNORECASE)
    width_pattern = re.compile(r'^\s*DECAY\s+9000001\s+([^\s#]+)', re.IGNORECASE)

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


def process_one_root(root_file, base_dir):
    """
    对单个 root 文件提取：
    sample, ma, fa, width, total_LLPs, LLP_1-200cm。
    """
    rel_path = os.path.relpath(root_file, base_dir)
    parts = rel_path.split(os.sep)
    sample = parts[0] if len(parts) > 0 else ""

    run_dir = os.path.dirname(root_file)
    banner_file = os.path.join(run_dir, "run_01_tag_1_banner.txt")

    ma, fa, width = parse_banner_file(banner_file)
    total_llps, total_events, llp_1_200cm, tree_path, diagnostics = get_total_llps_and_l3d_window(
        root_file,
        TARGET_BRANCH,
    )

    result = {
        "sample": sample,
        "ma": ma,
        "fa": fa,
        "width": width,
        "total_LLPs": total_llps,
        LLP_WINDOW_COLUMN: llp_1_200cm,
        "total_events": total_events,
        "root_file": root_file,
        "banner_file": banner_file,
        "tree": tree_path,
        "status": "ok",
    }
    result.update(diagnostics)
    return result


def make_error_result(root_file, error):
    """
    Build one error row. Kept as a top-level function so it is safe with multiprocessing.
    """
    return {
        "sample": "",
        "ma": np.nan,
        "fa": np.nan,
        "width": np.nan,
        "total_LLPs": np.nan,
        LLP_WINDOW_COLUMN: np.nan,
        "total_events": np.nan,
        "root_file": root_file,
        "banner_file": os.path.join(os.path.dirname(root_file), "run_01_tag_1_banner.txt"),
        "tree": "",
        "status": f"error: {error}",
    }


def process_one_root_safe(task):
    """
    Multiprocessing wrapper.

    Returns (order, result), so the final CSV can keep the deterministic sorted root-file order
    even though files are processed in parallel.
    """
    order, root_file, base_dir = task
    try:
        result = process_one_root(root_file, base_dir)
    except Exception as e:
        result = make_error_result(root_file, e)
    result["_order"] = order
    return order, result


def main():
    root_files = find_root_files(BASE_DIR)

    if not root_files:
        print("没有找到任何位于 */Events/run_01/*.root 的文件")
        return

    n_workers = max(1, int(N_WORKERS))
    n_workers = min(n_workers, len(root_files))

    print(f"共找到 {len(root_files)} 个 root 文件\n")
    print(f"L3D cut: {L3D_MIN_CM} cm < L3D_cm < {L3D_MAX_CM} cm")
    print(f"ALP_ID = {ALP_ID}")
    print(f"新增输出列: {LLP_WINDOW_COLUMN}")
    print(f"并行进程数: {n_workers}\n")

    tasks = [(i, root_file, BASE_DIR) for i, root_file in enumerate(root_files)]
    results = []
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_one_root_safe, task): task[1] for task in tasks}

        for future in as_completed(futures):
            root_file = futures[future]
            done += 1
            try:
                _, result = future.result()
            except Exception as e:
                # This should be rare because process_one_root_safe catches normal file-level errors.
                result = make_error_result(root_file, e)
                result["_order"] = root_files.index(root_file)

            results.append(result)

            if done == 1 or done % 50 == 0 or done == len(root_files):
                print(f"Processed {done}/{len(root_files)} files")

    results.sort(key=lambda row: row.get("_order", 10**18))
    for row in results:
        row.pop("_order", None)

    df = pd.DataFrame(results)

    # 你真正关心的主表：保留原来的列，并新增 LLP_1-200cm。
    main_cols = ["sample", "ma", "fa", "width", "total_LLPs", LLP_WINDOW_COLUMN]
    df_main = df[main_cols].copy()

    # 保存完整表和精简表。
    full_csv = OUTPUT_CSV
    slim_csv = os.path.join(BASE_DIR, "alp_llp_summary_slim.csv")

    df.to_csv(full_csv, index=False)
    df_main.to_csv(slim_csv, index=False)

    print("\n已保存：")
    print(full_csv)
    print(slim_csv)

    ok_df = df[df["status"] == "ok"]
    if not ok_df.empty:
        print("\n前几行结果预览：")
        print(ok_df[main_cols].head().to_string(index=False))

    failed_df = df[df["status"] != "ok"]
    if not failed_df.empty:
        print("\n以下文件处理失败：")
        print(failed_df[["root_file", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
