#!/usr/bin/env python3
"""
Day 1 产出 · 参数扫描优化器 (Surrogate-Based Optimization)
用法:
  python3 param_optimizer.py --csv scoring_telemetry.csv --n-trials 500
  python3 param_optimizer.py --baseline-score 89.2 --mode grid

不依赖真实仿真，而是基于遥测数据构建"代理打分器"，快速筛出 Top 参数组合，
再回真实环境验证。节省你大量仿真/真机飞行时间。
"""

import argparse
import csv
import json
import math
import random
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import interpolate, stats


BASELINE = {
    "lookahead_distance": 0.40,
    "lookahead_start_distance": 0.25,
    "lookahead_end_distance": 0.70,
    "lookahead_ramp_ratio": 0.20,
    "straight_lookahead_distance": 0.95,
    "turn_lookahead_distance": 0.20,
    "turn_buffer_distance": 0.45,
    "pass_radius": 0.50,
    "max_track_error": 1.5,
    "recover_track_error": 0.8,
    "takeoff_accept_radius": 0.12,
    "land_descent_rate": 0.30,
}

# 评分规则 (来自 visualization_scoring.py)
E_FULL, E_ZERO = 0.0, 0.50
T_FULL, T_ZERO = 10.0, 60.0


def score(rms, t, valid=True):
    acc = 100 * (E_ZERO - rms) / (E_ZERO - E_FULL)
    ts = 100 * (T_ZERO - t) / (T_ZERO - T_FULL)
    acc = max(0, min(100, acc))
    ts = max(0, min(100, ts))
    return 0.5 * acc + 0.5 * ts if valid else 0.0


# ============================================================
# 核心：代理打分器 (Surrogate Scorer)
# 基于遥测数据统计，将参数变化 -> 误差/时间 的变化做启发式映射
# 不完美，但 48h 内能给正确的优化方向
# ============================================================
class SurrogateDroneScorer:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.tr = tr = df[df["stage"] == "TRACKING"].sort_values("ros_time_s").dropna(subset=["track_error_m"]).reset_index(drop=True)
        last = tr.iloc[-1]
        self.baseline_rms = float(last["rms_error_m"])
        self.baseline_time = float(last["flight_time_s"])
        self.baseline_score = score(self.baseline_rms, self.baseline_time)

        # 分析各段
        self.total_len = float(tr["path_progress_m"].max())
        tr["seg_bin"] = pd.cut(tr["path_progress_ratio"], bins=5, labels=False)
        self.seg_err = tr.groupby("seg_bin")["track_error_m"].mean().values  # 5段
        # 门位置 (5段，中间的 1,2,3 段对应 3 个门附近)
        self.gate_err = self.seg_err[1:4].mean()
        self.straight_err = (self.seg_err[0] + self.seg_err[4]).mean() / 2

        # 估计转弯点数 (中间航点处) 与 直行段的比例
        self.turn_ratio = 0.35
        self.straight_ratio = 0.65

    def predict(self, p):
        """p is a param dict -> (predicted_rms, predicted_time, predicted_score)"""
        bl_rms = self.baseline_rms
        bl_time = self.baseline_time

        # ---- RMS 误差变化 ----
        err_factor = 1.0
        # lookahead_distance: 越大越冲越大 -> 误差↑, 时间↓
        lda_ratio = p["lookahead_distance"] / BASELINE["lookahead_distance"]
        err_factor *= 0.6 + 0.4 * lda_ratio  # [0.6~1.4] 区间映射
        # straight_lookahead: 影响直道段
        sla_r = p["straight_lookahead_distance"] / BASELINE["straight_lookahead_distance"]
        err_factor *= (0.85 + 0.15 * sla_r) ** self.straight_ratio
        # turn_lookahead: 越小转弯越紧 -> 门附近误差↓
        tla_r = p["turn_lookahead_distance"] / BASELINE["turn_lookahead_distance"]
        turn_err_factor = (0.8 + 0.2 * tla_r) ** self.turn_ratio
        err_factor *= turn_err_factor
        # turn_buffer: 越大提前减速 -> 转弯误差↓
        tbuf_r = p["turn_buffer_distance"] / BASELINE["turn_buffer_distance"]
        err_factor *= (1.05 - 0.05 * tbuf_r) ** self.turn_ratio
        # pass_radius: 越小越严格 -> 早期过门误差↓但时间↑
        pr_r = p["pass_radius"] / BASELINE["pass_radius"]
        err_factor *= (0.95 + 0.05 * pr_r)

        pred_rms = bl_rms * err_factor
        pred_rms = max(0.005, pred_rms)

        # ---- 飞行时间变化 ----
        time_factor = 1.0
        # lookahead_end_distance: 越大越"冲" -> 时间↓
        led_r = p["lookahead_end_distance"] / BASELINE["lookahead_end_distance"]
        time_factor *= (1.08 - 0.08 * led_r)
        # straight_lookahead: 直行越快
        time_factor *= (1.05 - 0.05 * sla_r) ** self.straight_ratio
        # turn_lookahead 小 -> 内道 -> 路径短 -> 时间↓
        time_factor *= (0.97 + 0.03 * tla_r) ** self.turn_ratio
        # pass_radius 小 -> 不绕行 -> 时间略↓
        time_factor *= (0.985 + 0.015 * pr_r)
        # takeoff_accept_radius: 越大 -> 起飞段越快
        tar_r = p["takeoff_accept_radius"] / BASELINE["takeoff_accept_radius"]
        time_factor *= (1.0 - 0.01 * math.log1p(max(0, tar_r - 1.0)))
        # land_descent_rate: 越大 -> 降落越快
        ldr_r = p["land_descent_rate"] / BASELINE["land_descent_rate"]
        time_factor *= (1.0 - 0.005 * (ldr_r - 1.0)) if ldr_r >= 1.0 else (1.0 + 0.01 * (1.0 - ldr_r))

        pred_time = bl_time * time_factor
        pred_time = max(T_FULL, min(T_ZERO, pred_time))

        # ---- 有效性检查: 触发恢复模式太多视为invalid风险 ----
        recover_risk = 0.0
        maxterr_expected = pred_rms * 3.0 + 0.1
        if p["max_track_error"] < maxterr_expected * 0.9:
            recover_risk += 0.1
        if p["recover_track_error"] > p["max_track_error"] * 0.9:
            recover_risk += 0.05
        valid = recover_risk < 0.25

        pred_score = score(pred_rms, pred_time, valid)
        return round(pred_rms, 5), round(pred_time, 3), round(pred_score, 3), round(recover_risk, 3)


# ============================================================
# 参数空间
# ============================================================
PARAM_SPACE = {
    "lookahead_distance": (0.25, 0.75),
    "lookahead_start_distance": (0.15, 0.45),
    "lookahead_end_distance": (0.50, 1.20),
    "lookahead_ramp_ratio": (0.10, 0.40),
    "straight_lookahead_distance": (0.60, 1.20),
    "turn_lookahead_distance": (0.10, 0.30),
    "turn_buffer_distance": (0.30, 0.80),
    "pass_radius": (0.30, 0.70),
    "max_track_error": (1.20, 2.20),
    "recover_track_error": (0.50, 1.20),
    "takeoff_accept_radius": (0.08, 0.25),
    "land_descent_rate": (0.20, 0.60),
}


def random_params():
    p = {}
    for k, (lo, hi) in PARAM_SPACE.items():
        p[k] = round(random.uniform(lo, hi), 4)
    # 一致性约束: start < base < end, recover < max
    p["lookahead_start_distance"] = min(p["lookahead_start_distance"], p["lookahead_distance"] * 0.9)
    p["lookahead_end_distance"] = max(p["lookahead_end_distance"], p["lookahead_distance"] * 1.1)
    p["recover_track_error"] = min(p["recover_track_error"], p["max_track_error"] * 0.8)
    return p


def local_search(center, scorer, n_neighbors=50, step=0.08):
    best_p, best_s = center, -1
    for _ in range(n_neighbors):
        cand = {}
        for k, (lo, hi) in PARAM_SPACE.items():
            c = center[k]
            delta = (hi - lo) * step * random.gauss(0, 1)
            cand[k] = round(max(lo, min(hi, c + delta)), 4)
        cand["lookahead_start_distance"] = min(cand["lookahead_start_distance"], cand["lookahead_distance"] * 0.9)
        cand["lookahead_end_distance"] = max(cand["lookahead_end_distance"], cand["lookahead_distance"] * 1.1)
        cand["recover_track_error"] = min(cand["recover_track_error"], cand["max_track_error"] * 0.8)
        _, _, s, risk = scorer.predict(cand)
        if s > best_s and risk < 0.15:
            best_s = s
            best_p = cand
    return best_p, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="scoring_telemetry.csv")
    ap.add_argument("--n-trials", type=int, default=500)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["random", "grid", "both"], default="both")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"🧠 初始化代理打分器: {args.csv}")
    scorer = SurrogateDroneScorer(args.csv)
    print(f"   基线: RMS={scorer.baseline_rms:.4f}m, 时间={scorer.baseline_time:.2f}s, 分数={scorer.baseline_score:.2f}")

    results = []

    # --- 1. 随机+局部搜索 ---
    if args.mode in ("random", "both"):
        print(f"\n🎲 随机搜索 {args.n_trials} 组 + 局部搜索...")
        for i in range(args.n_trials):
            p = random_params()
            rms, t, s, risk = scorer.predict(p)
            if s > scorer.baseline_score:
                p_refined, s_refined = local_search(p, scorer, n_neighbors=40)
                rms_r, t_r, _, _ = scorer.predict(p_refined)
                results.append((s_refined, rms_r, t_r, p_refined))
            else:
                results.append((s, rms, t, p))

    # --- 2. 粗网格 (只搜 4 个最敏感参数) ---
    if args.mode in ("grid", "both"):
        print("🔲 粗网格搜索关键参数...")
        keys = ["lookahead_distance", "straight_lookahead_distance", "turn_lookahead_distance", "turn_buffer_distance"]
        grids = [np.linspace(*PARAM_SPACE[k], 5) for k in keys]
        import itertools
        for vals in itertools.product(*grids):
            p = deepcopy(BASELINE)
            for k, v in zip(keys, vals):
                p[k] = round(float(v), 4)
            rms, t, s, risk = scorer.predict(p)
            if risk < 0.2:
                results.append((s, rms, t, p))

    # 去重 + 排序
    seen = set()
    uniq = []
    for s, r, t, p in results:
        key = tuple(round(v, 3) for v in p.values())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((s, r, t, p))
    uniq.sort(key=lambda x: -x[0])

    print(f"\n🏆 Top-{args.top_k} 参数组合 (基线分数 {scorer.baseline_score:.2f})")
    print(f"{'Rank':>4}  {'Score':>6}  {'RMS':>7}  {'Time':>6}  Δ")
    top_out = []
    for rank, (s, rms, t, p) in enumerate(uniq[: args.top_k], 1):
        delta = s - scorer.baseline_score
        arrow = "🔺" if delta > 0 else "▫️"
        print(f"{rank:>4}  {s:>6.2f}  {rms:>7.4f}  {t:>6.2f}  {arrow} +{delta:+.2f}")
        top_out.append({
            "rank": rank,
            "score": s,
            "rms_error_m": rms,
            "flight_time_s": t,
            "score_delta_vs_baseline": round(delta, 3),
            "params": p,
        })

    # --- 敏感性分析: 每个参数对分数的影响 ---
    print("\n📈 参数敏感性分析 (每个参数拉到边界后的分数差)")
    sensitivity = {}
    for k, (lo, hi) in PARAM_SPACE.items():
        p_lo = deepcopy(BASELINE); p_lo[k] = lo; _, _, s_lo, _ = scorer.predict(p_lo)
        p_hi = deepcopy(BASELINE); p_hi[k] = hi; _, _, s_hi, _ = scorer.predict(p_hi)
        sensitivity[k] = round(max(abs(s_lo - scorer.baseline_score), abs(s_hi - scorer.baseline_score)), 3)
    for k, v in sorted(sensitivity.items(), key=lambda x: -x[1]):
        bar = "█" * int(v * 4)
        print(f"   {k:>32s}  {v:+.3f} {bar}")

    # --- 保存结果 ---
    out = {
        "baseline": {
            "params": BASELINE,
            "rms_error_m": scorer.baseline_rms,
            "flight_time_s": scorer.baseline_time,
            "score": scorer.baseline_score,
        },
        "sensitivity": sensitivity,
        "top_candidates": top_out,
    }
    with open("optimization_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # 同时输出一个能直接复制到 offboard_ros2.py 的 TOP1 参数片段
    best_p = top_out[0]["params"]
    snippet_lines = ["        # ============================================================",
                     "        # AI 优化参数 (来自 param_optimizer.py Top-1)",
                     "        # 预测分数: {:.2f} (基线 {:.2f}, Δ{:+.2f})".format(top_out[0]["score"], scorer.baseline_score, top_out[0]["score_delta_vs_baseline"]),
                     "        # ============================================================",]
    for k in BASELINE.keys():
        if k in best_p:
            snippet_lines.append(f"        self.{k} = {best_p[k]}")
    with open("best_params_snippet.py.txt", "w") as f:
        f.write("\n".join(snippet_lines) + "\n")

    print(f"\n✅ 已保存:")
    print(f"   · optimization_results.json   → Top{args.top_k} + 敏感性分析")
    print(f"   · best_params_snippet.py.txt → Top1 参数，复制粘贴到 offboard_ros2.py 即可")


if __name__ == "__main__":
    main()
