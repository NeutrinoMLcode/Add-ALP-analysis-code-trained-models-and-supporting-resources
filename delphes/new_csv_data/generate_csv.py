#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import uproot

BASE_DIR = "/home/wsy/MG5_aMC_v3_5_6_LLP/ALP_LLP"
TARGET_BRANCH = "LLPs_size"
OUTPUT_CSV = os.path.join(BASE_DIR, "alp_llp_summary.csv")


def find_root_files(base_dir):
    """
    查找所有位于 */Events/run_01/*.root 的文件
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
    在 root 文件中查找包含指定 branch 的 TTree
    返回 tree_path，找不到则返回 None
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


def get_total_llps(root_file, branch_name="LLPs_size"):
    """
    读取 root 文件中的 LLPs_size，并返回 total_LLPs = sum(LLPs_size)
    """
    tree_path = find_tree_with_branch(root_file, branch_name)
    if tree_path is None:
        raise RuntimeError(f"branch '{branch_name}' not found in {root_file}")

    with uproot.open(root_file) as f:
        tree = f[tree_path]
        sizes = tree[branch_name].array(library="np")

    total_llps = int(np.sum(sizes, dtype=np.int64))
    total_events = int(len(sizes))
    return total_llps, total_events, tree_path


def parse_banner_file(banner_file):
    """
    从 run_01_tag_1_banner.txt 中提取:
    - ma    : 9000001 10 # ma 这一行中的第二列
    - fa    : 1 1000000 # fa 这一行中的第二列
    - width : DECAY 9000001 1.700000e-16 这一行中的第三列
    """
    if not os.path.exists(banner_file):
        raise FileNotFoundError(f"banner file not found: {banner_file}")

    ma = None
    fa = None
    width = None

    # 更稳妥的正则：用注释标签区分 ma 和 fa
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
    对单个 root 文件提取:
    sample, ma, fa, width, total_LLPs
    """
    rel_path = os.path.relpath(root_file, base_dir)
    parts = rel_path.split(os.sep)
    sample = parts[0] if len(parts) > 0 else ""

    run_dir = os.path.dirname(root_file)
    banner_file = os.path.join(run_dir, "run_01_tag_1_banner.txt")

    ma, fa, width = parse_banner_file(banner_file)
    total_llps, total_events, tree_path = get_total_llps(root_file, TARGET_BRANCH)

    return {
        "sample": sample,
        "ma": ma,
        "fa": fa,
        "width": width,
        "total_LLPs": total_llps,
        "total_events": total_events,
        "root_file": root_file,
        "banner_file": banner_file,
        "tree": tree_path,
        "status": "ok"
    }


def main():
    root_files = find_root_files(BASE_DIR)

    if not root_files:
        print("没有找到任何位于 */Events/run_01/*.root 的文件")
        return

    print(f"共找到 {len(root_files)} 个 root 文件\n")

    results = []
    for i, root_file in enumerate(root_files, 1):
        print(f"[{i}/{len(root_files)}] Processing: {root_file}")
        try:
            result = process_one_root(root_file, BASE_DIR)
        except Exception as e:
            result = {
                "sample": "",
                "ma": np.nan,
                "fa": np.nan,
                "width": np.nan,
                "total_LLPs": np.nan,
                "total_events": np.nan,
                "root_file": root_file,
                "banner_file": os.path.join(os.path.dirname(root_file), "run_01_tag_1_banner.txt"),
                "tree": "",
                "status": f"error: {e}"
            }
        results.append(result)

    df = pd.DataFrame(results)

    # 你真正关心的主表
    main_cols = ["sample", "ma", "fa", "width", "total_LLPs"]
    df_main = df[main_cols].copy()

    # 保存完整表和精简表
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
