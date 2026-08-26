#!/usr/bin/env python3
"""
Day 1 产出 · 遥测分析 + LLM参数调优助手
用法: python3 telemetry_analyzer.py [csv_path]
  - 分析 scoring_telemetry.csv，自动定位误差瓶颈
  - 生成 LLM 友好的结构化报告(可直接复制给大模型问建议)
  - 无 LLM API 也能跑，内置启发式调参建议
"""

import csv
import math
import sys
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = "scoring_telemetry.csv" if len(sys.argv) < 2 else sys.argv[1]

# === 评分规则 (与 visualization_scoring.py 保持一致) ===
ERROR_FULL = 0.0
ERROR_ZERO = 0.50
TIME_FULL = 10.0
TIME_ZERO = 60.0
ACC_W = 0.50
TIME_W = 0.50

# === 当前控制器的默认参数 (来自 offboard_ros2.py) ===
BASELINE_PARAMS = {
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
    "rate_hz": 30.0,
}


def compute_scores(rms_err, flight_time, run_valid=True):
    err_span = ERROR_ZERO - ERROR_FULL
    acc = 100.0 * (ERROR_ZERO - rms_err) / err_span
    acc = max(0.0, min(100.0, acc))
    t_span = TIME_ZERO - TIME_FULL
    ts = 100.0 * (TIME_ZERO - flight_time) / t_span
    ts = max(0.0, min(100.0, ts))
    final = ACC_W * acc + TIME_W * ts
    if not run_valid:
        final = 0.0
    return round(acc, 2), round(ts, 2), round(final, 2)


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    tracking = df[df["stage"] == "TRACKING"].copy().sort_values("ros_time_s").reset_index(drop=True)
    tracking = tracking.dropna(subset=["track_error_m", "x_m", "y_m", "z_m"])
    return df, tracking


def bottleneck_analysis(tracking):
    """自动分析哪段路径、哪项因素丢分最多"""
    report = {}

    # --- 最终指标 ---
    last = tracking.iloc[-1]
    acc_s, time_s, final_s = compute_scores(last["rms_error_m"], last["flight_time_s"], bool(last.get("run_valid", True)))
    report["final_scores"] = {"accuracy": acc_s, "time": time_s, "final": final_s}
    report["rms_error_m"] = round(float(last["rms_error_m"]), 4)
    report["flight_time_s"] = round(float(last["flight_time_s"]), 3)
    report["rings_passed"] = f"{int(last['next_gate'])}/{int(last['rings_total'])}"

    # --- 精度 vs 时间 哪项拖后腿? ---
    if acc_s < time_s:
        culprit = "accuracy"
        margin = time_s - acc_s
    else:
        culprit = "time"
        margin = acc_s - time_s
    report["main_culprit"] = culprit
    report["culprit_margin"] = round(margin, 2)

    # --- 按路径进度 4 段统计误差 ---
    bins = [0, 0.25, 0.5, 0.75, 1.01]
    labels = ["S1(0-25%): 起飞→门1", "S2(25-50%): 门1→门2", "S3(50-75%): 门2→门3", "S4(75-100%): 门3→终点"]
    tracking["seg"] = pd.cut(tracking["path_progress_ratio"], bins=bins, labels=labels, include_lowest=True)
    seg = tracking.groupby("seg", observed=True).agg(
        err_mean=("track_error_m", "mean"),
        err_max=("track_error_m", "max"),
        err_p95=("track_error_m", lambda x: x.quantile(0.95)),
        z_mean=("z_m", "mean"),
        z_std=("z_m", "std"),
        n=("track_error_m", "count"),
    ).round(4)
    report["segment_error"] = seg.to_dict(orient="index")
    worst_seg = seg["err_mean"].idxmax()
    report["worst_segment"] = worst_seg
    report["worst_seg_error"] = round(float(seg.loc[worst_seg, "err_mean"]), 4)

    # --- Z 高度稳定性 ---
    z_mean = tracking["z_m"].mean()
    z_std = tracking["z_m"].std()
    report["z_stats"] = {"mean_m": round(z_mean, 4), "std_m": round(z_std, 4), "target_m": 2.0, "bias_m": round(z_mean - 2.0, 4)}

    # --- 误差与路径进度的相关性 (越大说明越到后面越飘) ---
    corr_progress_err = stats.pearsonr(tracking["path_progress_ratio"].values, tracking["track_error_m"].values)[0]
    report["corr_progress_vs_error"] = round(float(corr_progress_err), 3)

    # --- 检测误差尖峰 (>= P99) 位置 ---
    p99 = tracking["track_error_m"].quantile(0.99)
    spikes = tracking[tracking["track_error_m"] >= p99]
    report["spike_count_p99"] = int(len(spikes))
    report["spike_avg_progress_ratio"] = round(float(spikes["path_progress_ratio"].mean()), 3) if len(spikes) else 0.0

    return report


def heuristic_tips(report):
    """不依赖 LLM API，基于规则给出调参建议 (2天内能直接用)"""
    tips = []

    # 1. 精度拖后腿
    if report["main_culprit"] == "accuracy":
        tips.append(f"[精度优先] 精度分比时间分低 {report['culprit_margin']} 分 → 建议降低 straight_lookahead_distance (当前0.95 → 试 0.70~0.85)，牺牲一点速度换跟踪精度。")
    else:
        tips.append(f"[速度优先] 时间分比精度分低 {report['culprit_margin']} 分 → 建议增大 lookahead_end_distance (当前0.70 → 试 0.85~1.10)，飞得更'冲'。")

    # 2. 最坏段在 门附近 (S2/S3，门过门切换处)
    ws = report["worst_segment"]
    if "门1→门2" in ws or "门2→门3" in ws:
        tips.append(f"[转弯优化] 最大误差在 {ws} (err={report['worst_seg_error']}m) → 建议减小 turn_lookahead_distance (当前0.20 → 试 0.12~0.18)，增大 turn_buffer_distance (当前0.45 → 试 0.55~0.70)，提前切内道。")
    elif "起飞→门1" in ws:
        tips.append(f"[起飞阶段] 最大误差在 {ws} → 建议增大 takeoff_accept_radius (当前0.12 → 试 0.18~0.25)，起飞后尽快切入跟踪。")
    else:
        tips.append(f"[终点阶段] 最大误差在 {ws} → 建议减小 lookahead_end_distance 或 增大 final_accept_radius。")

    # 3. Z 轴偏差大
    z_bias = abs(report["z_stats"]["bias_m"])
    z_std = report["z_stats"]["std_m"]
    if z_bias > 0.05 or z_std > 0.05:
        tips.append(f"[Z轴不稳] 高度偏差={report['z_stats']['bias_m']}m, 波动std={z_std}m → 检查 PX4 垂控增益 (MPC_Z_P, MPC_Z_VEL_P)，或减小 OFFBOARD 降落切换阈值 land_switch_to_auto_land_z。")

    # 4. 误差随时间累积
    if report["corr_progress_vs_error"] > 0.3:
        tips.append(f"[漂移] 误差与路径进度相关系数={report['corr_progress_vs_error']} (越飞越飘) → 可能是积分器饱和或位姿延迟，考虑减小 pass_radius (当前0.50→0.40) 让门穿越更紧。")
    elif report["corr_progress_vs_error"] < -0.3:
        tips.append(f"[起飞抖动] 相关系数负 = 起飞快后面稳 → 增大 takeoff_staging_radius 或 延长 takeoff_approach 段。")

    # 5. 尖峰多
    if report["spike_count_p99"] >= 10:
        tips.append(f"[抖动抑制] P99尖峰有 {report['spike_count_p99']} 个 → 建议调大 max_track_error 触发恢复的阈值 (当前1.5→1.8)，同时收紧 recover_track_error (当前0.8→0.6)。")

    return tips


def build_llm_prompt(report, tips):
    """生成一段可以直接扔给 Qwen/GPT 的结构化 prompt"""
    return f"""
你是无人机竞速控制参数调优专家。以下是上一轮飞行的遥测分析报告：

【最终成绩】
- 精度分: {report['final_scores']['accuracy']}/100
- 时间分: {report['final_scores']['time']}/100
- **总分: {report['final_scores']['final']}/100**
- RMS 误差: {report['rms_error_m']} m
- 飞行时间: {report['flight_time_s']} s
- 过门情况: {report['rings_passed']}

【核心诊断】
- 主要拖后腿维度: **{report['main_culprit']}** (领先/落后 {report['culprit_margin']} 分)
- 误差最大路径段: **{report['worst_segment']}** (平均误差 {report['worst_seg_error']} m)
- Z轴: 均值 {report['z_stats']['mean_m']}m (目标2.0m, 偏差 {report['z_stats']['bias_m']}m), 波动std {report['z_stats']['std_m']}m
- 误差随路径累积相关系数: {report['corr_progress_vs_error']} (>0.3表示越飞越飘)
- P99以上误差尖峰: {report['spike_count_p99']} 个，集中在进度 {report['spike_avg_progress_ratio']} 附近

【分段误差明细】
{json.dumps(report['segment_error'], indent=2, ensure_ascii=False)}

【当前基线参数】
{json.dumps(BASELINE_PARAMS, indent=2)}

【启发式初步建议】
{chr(10).join('- '+t for t in tips)}

请基于以上信息，给出：
1. 按优先级排序的 Top 5 参数调整建议（每一项都说明: 参数名、原值→建议值、为什么这么改）
2. 给出一个「保守方案」和「激进方案」两个参数组合 JSON，方便直接写入 offboard_ros2.py
3. 如果有 3 轮参数迭代计划，请说明每一轮试什么、预期指标如何变化
"""


def save_artifacts(report, tips, prompt):
    with open("analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open("llm_tuning_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"\n✅ 已保存文件:")
    print(f"   · analysis_report.json        → 结构化报告")
    print(f"   · llm_tuning_prompt.txt       → 复制给 LLM 的 prompt")


def main():
    print(f"📊 读取遥测数据: {CSV_PATH}")
    _, tracking = load_and_clean(CSV_PATH)
    print(f"   跟踪阶段样本: {len(tracking)} 行")

    print("\n🔍 进行瓶颈分析...")
    report = bottleneck_analysis(tracking)

    print("\n" + "="*60)
    print("📋 遥测诊断报告")
    print("="*60)
    s = report["final_scores"]
    print(f"  [成绩] 精度 {s['accuracy']} | 时间 {s['time']} | **总分 {s['final']}/100**")
    print(f"  [指标] RMS误差={report['rms_error_m']}m | 时间={report['flight_time_s']}s | 过门={report['rings_passed']}")
    print(f"  [短板] {report['main_culprit']} 拖后腿 (差距 {report['culprit_margin']} 分)")
    print(f"  [重灾] {report['worst_segment']} · 平均err {report['worst_seg_error']}m")
    print(f"  [Z轴] 偏差 {report['z_stats']['bias_m']}m · 波动 {report['z_stats']['std_m']}m")
    print(f"  [漂移] 进度×误差相关系数 {report['corr_progress_vs_error']}")

    print("\n💡 启发式调参建议 (无LLM也能用):")
    tips = heuristic_tips(report)
    for i, t in enumerate(tips, 1):
        print(f"  {i}. {t}")

    prompt = build_llm_prompt(report, tips)
    save_artifacts(report, tips, prompt)
    print("\n🤖 → 把 llm_tuning_prompt.txt 内容粘贴给任意大模型，就能拿到AI优化的参数方案！")


if __name__ == "__main__":
    main()
