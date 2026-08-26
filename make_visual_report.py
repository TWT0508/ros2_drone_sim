#!/usr/bin/env python3
"""
Day 2 上午 · 一键可视化报告
生成:
  · report_fig1_telemetry_analysis.png → 遥测误差/轨迹/分段分析
  · report_fig2_sensitivity.png       → 参数敏感性条形图
  · report_fig3_optimization_gain.png → 基线 vs Top10 对比

用法: python3 make_visual_report.py
"""

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体兜底 (没装的话就用英文标签)
try:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# === 颜色 (和dynamic-ui一致) ===
C_BASELINE = "#6b7280"
C_ACC = "#7c3aed"
C_TIME = "#0ea5e9"
C_FINAL = "#10b981"
C_RMS = "#f59e0b"
SERIES = ["#7c3aed", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444"]

CSV = "scoring_telemetry.csv"
OPT_JSON = "optimization_results.json"


def load_data():
    df = pd.read_csv(CSV)
    tr = df[df["stage"] == "TRACKING"].copy().sort_values("ros_time_s").reset_index(drop=True)
    tr = tr.dropna(subset=["track_error_m", "x_m", "y_m", "z_m"])
    with open(OPT_JSON, encoding="utf-8") as f:
        opt = json.load(f)
    return df, tr, opt


def expand_path_from_waypoints():
    # 和 offboard_ros2.py 保持一致的路径展开
    wps = np.array([[0.0, 0.0, 2.0, 0.0],
                    [2.0, 0.0, 2.0, 45.0],
                    [2.0, 2.0, 2.0, 135.0],
                    [0.0, 2.0, 2.0, -135.0],
                    [0.0, 0.5, 2.0, 0.0]])
    gates_pre, gates_post = 0.8, 0.9
    pts = [wps[0, :3]]
    gates_xy = []
    for i in range(1, len(wps) - 1):
        c = wps[i, :3]
        yaw = math.radians(wps[i, 3])
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        pts.extend([c - gates_pre * fwd, c, c + gates_post * fwd])
        gates_xy.append(c[:2])
    pts.append(wps[-1, :3])
    return np.array(pts), np.array(gates_xy)


def fig1_telemetry(tr):
    path, gates_xy = expand_path_from_waypoints()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=120)
    fig.suptitle("Drone Race · Telemetry Deep-Dive (Baseline 89.2/100)", fontsize=14, fontweight="bold")

    # (0,0) 路径进度 vs 误差
    ax = axes[0, 0]
    prog = tr["path_progress_ratio"].values
    err = tr["track_error_m"].values
    rms = tr["rms_error_m"].values
    ax.scatter(prog, err, s=6, c=C_ACC, alpha=0.5, label="Instant error")
    id_sort = np.argsort(prog)
    ax.plot(prog[id_sort], rms[id_sort], c=C_RMS, lw=2, label="RMS (cumul.)")
    ax.axhline(0.50, ls="--", lw=1, c="#ef4444", alpha=0.7, label="Zero-score line (0.5m)")
    ax.axvline(0.25, ls=":", lw=0.8, c="gray")
    ax.axvline(0.50, ls=":", lw=0.8, c="gray")
    ax.axvline(0.75, ls=":", lw=0.8, c="gray")
    for gi, gx in enumerate([0.2, 0.5, 0.8], 1):
        ax.annotate(f"Gate {gi}", xy=(gx, 0), xytext=(gx, err.max() * 0.9),
                    ha="center", fontsize=9, color="#ef4444", alpha=0.7,
                    arrowprops=dict(arrowstyle="->", color="#ef4444", lw=0.8, alpha=0.5))
    ax.set_xlabel("Path Progress (0→1)")
    ax.set_ylabel("Track error [m]")
    ax.set_title("Error over path")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # (0,1) 分段误差柱状图
    ax = axes[0, 1]
    bins = [0, 0.25, 0.5, 0.75, 1.01]
    labels = ["Takeoff\n→ Gate1", "Gate1\n→ Gate2", "Gate2\n→ Gate3", "Gate3\n→ Finish"]
    tr["seg"] = pd.cut(tr["path_progress_ratio"], bins=bins, labels=labels, include_lowest=True)
    grp = tr.groupby("seg", observed=True)["track_error_m"]
    means = grp.mean().values
    p95s = grp.quantile(0.95).values
    maxs = grp.max().values
    xs = np.arange(len(labels))
    w = 0.27
    ax.bar(xs - w, means, w, label="Mean", color=C_ACC, alpha=0.85)
    ax.bar(xs, p95s, w, label="P95", color=C_TIME, alpha=0.85)
    ax.bar(xs + w, maxs, w, label="Max", color="#ef4444", alpha=0.75)
    for i in range(len(labels)):
        ax.text(xs[i] - w, means[i] + 0.003, f"{means[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Error [m]")
    ax.set_title("Segment-wise error breakdown")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (1,0) 俯视图: 参考路径 vs 真实轨迹 vs 门
    ax = axes[1, 0]
    ax.plot(path[:, 0], path[:, 1], "o--", ms=5, lw=2, c=C_ACC, alpha=0.7, label="Ref path")
    ax.plot(tr["x_m"], tr["y_m"], lw=2.2, c=C_TIME, alpha=0.9, label="Actual flight")
    for gi, (gx, gy) in enumerate(gates_xy, 1):
        circ = plt.Circle((gx, gy), 0.5, fill=False, lw=2.2, color="#ef4444", alpha=0.7)
        ax.add_artist(circ)
        ax.scatter(gx, gy, 100, marker="x", c="#ef4444", lw=2)
        ax.text(gx, gy + 0.6, f"Gate {gi}", ha="center", fontsize=10, color="#ef4444", fontweight="bold")
    ax.scatter(tr["x_m"].iloc[0], tr["y_m"].iloc[0], 80, c=C_FINAL, marker="D", zorder=5, label="Start")
    ax.scatter(tr["x_m"].iloc[-1], tr["y_m"].iloc[-1], 80, c="#ef4444", marker="*", zorder=5, label="End")
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Top view: Ref path vs Actual")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,1) 高度随时间
    ax = axes[1, 1]
    t = tr["flight_time_s"].values
    z = tr["z_m"].values
    yaw = tr["yaw_deg"].values
    ax.plot(t, z, c=C_ACC, lw=2, label="Altitude Z")
    ax.axhline(2.0, ls="--", lw=1.2, c=C_RMS, label="Target Z=2.0m")
    ax.set_xlabel("Flight time [s]")
    ax.set_ylabel("Z [m]", color=C_ACC)
    ax.tick_params(axis="y", labelcolor=C_ACC)
    ax2 = ax.twinx()
    ax2.plot(t, yaw, c=C_TIME, lw=1.2, alpha=0.8, label="Yaw (deg)")
    ax2.set_ylabel("Yaw [°]", color=C_TIME)
    ax2.tick_params(axis="y", labelcolor=C_TIME)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")
    ax.set_title("Altitude & Yaw profile")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = "report_fig1_telemetry_analysis.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def fig2_sensitivity(opt):
    sens = opt["sensitivity"]
    items = sorted(sens.items(), key=lambda x: -x[1])
    keys, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    colors = [SERIES[min(i // 2, len(SERIES) - 1)] for i in range(len(keys))]
    bars = ax.barh(range(len(keys)), vals, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=9)
    ax.invert_yaxis()
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"+{v:.2f} pts", va="center", fontsize=9, fontweight="bold", color=colors[i])
    ax.set_xlabel("Max score delta when parameter swept to bounds [pts]")
    ax.set_title("Parameter Sensitivity Ranking\n(how much each param can impact the score, by itself)")
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(1.0, ls="--", lw=1, c=C_RMS, alpha=0.5, label="1 point gain threshold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = "report_fig2_sensitivity.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def fig3_gain(opt):
    bl = opt["baseline"]
    tops = opt["top_candidates"][:8]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    fig.suptitle("Surrogate Optimization · Baseline 89.2 vs Top-8 AI candidates",
                 fontsize=13, fontweight="bold")

    ranks = ["Baseline"] + [f"Top {t['rank']}" for t in tops]
    scores = [round(bl["score"], 2)] + [t["score"] for t in tops]
    colors = [C_BASELINE] + SERIES * 2

    # (0) Final Score
    ax = axes[0]
    bars = ax.bar(range(len(ranks)), scores, color=colors[:len(ranks)], alpha=0.85, edgecolor="white")
    ax.axhline(bl["score"], ls="--", lw=1.5, c=C_BASELINE, alpha=0.6, label="Baseline")
    for i, (b, s) in enumerate(zip(bars, scores)):
        delta = s - bl["score"]
        tag = f"{s:.1f}\nΔ{delta:+.1f}" if i > 0 else f"{s:.1f}"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, tag,
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(ranks, fontsize=8, rotation=20)
    ax.set_ylabel("Final score [/100]")
    ax.set_ylim(min(scores) - 2, max(scores) + 2)
    ax.set_title("🏁 Final score")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # (1) RMS error
    ax = axes[1]
    rms_list = [round(bl["rms_error_m"], 5)] + [t["rms_error_m"] for t in tops]
    bars = ax.bar(range(len(ranks)), rms_list, color=colors[:len(ranks)], alpha=0.85, edgecolor="white")
    ax.axhline(bl["rms_error_m"], ls="--", lw=1.5, c=C_BASELINE, alpha=0.6)
    for i, (b, v) in enumerate(zip(bars, rms_list)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(ranks, fontsize=8, rotation=20)
    ax.set_ylabel("RMS error [m]  (↓ better)")
    ax.set_title("📏 RMS tracking error")
    ax.grid(axis="y", alpha=0.3)

    # (2) Flight time
    ax = axes[2]
    t_list = [round(bl["flight_time_s"], 3)] + [t["flight_time_s"] for t in tops]
    bars = ax.bar(range(len(ranks)), t_list, color=colors[:len(ranks)], alpha=0.85, edgecolor="white")
    ax.axhline(bl["flight_time_s"], ls="--", lw=1.5, c=C_BASELINE, alpha=0.6)
    for i, (b, v) in enumerate(zip(bars, t_list)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                f"{v:.2f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(ranks, fontsize=8, rotation=20)
    ax.set_ylabel("Flight time [s]  (↓ faster)")
    ax.set_title("⚡ Flight time")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = "report_fig3_optimization_gain.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def main():
    print("🎨 加载数据...")
    _, tr, opt = load_data()
    print("  遥测跟踪行:", len(tr), "优化结果已加载")

    f1 = fig1_telemetry(tr)
    print(f"✅ 图1生成: {f1}")

    f2 = fig2_sensitivity(opt)
    print(f"✅ 图2生成: {f2}")

    f3 = fig3_gain(opt)
    print(f"✅ 图3生成: {f3}")

    bl = opt["baseline"]
    top1 = opt["top_candidates"][0]
    print("\n" + "=" * 60)
    print("📊 总结: 基线 vs Top-1")
    print("=" * 60)
    print(f"         基线              →   Top1 (代理预测)")
    print(f"  分数:  {bl['score']:.2f}            →   {top1['score']:.2f}   (Δ {top1['score_delta_vs_baseline']:+.2f})")
    print(f"  RMS:   {bl['rms_error_m']:.4f}m         →   {top1['rms_error_m']:.4f}m")
    print(f"  时间:  {bl['flight_time_s']:.2f}s          →   {top1['flight_time_s']:.2f}s")
    print(f"  敏感参数 Top3: {list(opt['sensitivity'].keys())[:3]}")


if __name__ == "__main__":
    main()
