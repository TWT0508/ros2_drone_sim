#!/usr/bin/env python3
"""
Day 2 储备 · 无人机竞速 Gymnasium RL 环境骨架
⚠️ 两天内不建议直接上训练，先把接口打通 + 跑通随机动作 smoke test 即可

用法:
    # smoke test (5个随机回合)
    python3 drone_rl_env.py --smoke 5

    # 接 Stable-Baselines3 训练 (需要你自己有 ROS2+PX4 仿真环境)
    # 见下方 train_with_sb3() 示例函数

Observation Space (15维):
    [pos_x, pos_y, pos_z,         # 当前位置 (m)
     vel_x, vel_y, vel_z,         # 当前速度 (m/s)  *用有限差分估算
     progress_s/total_len,       # 路径进度 (归一化0-1)
     track_error,                 # 到参考线距离 (m)
     lookahead_point_rel_x/y/z,   # 前瞻点相对无人机
     next_gate_rel_x/y/z]         # 下一个门相对无人机

Action Space (3维 Box [-1,1]，映射到实际目标位置增量):
    [dx, dy, dz]  →  target = current_pos + action_scale * [dx, dy, dz]

Reward (和评分强对齐):
    r = 0.5 * accuracy_component + 0.5 * speed_component - penalty_shaping
"""

import argparse
import math
import random
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# === 从 offboard_ros2.py 复制的几何工具 ===
# (保持独立，不依赖 rclpy)
DEFAULT_WAYPOINTS = [
    [0.0, 0.0, 2.0, 0.0],
    [2.0, 0.0, 2.0, 45.0],
    [2.0, 2.0, 2.0, 135.0],
    [0.0, 2.0, 2.0, -135.0],
    [0.0, 0.5, 2.0, 0.0],
]
GATE_PRE = 0.8
GATE_POST = 0.9


def build_expanded_path(wps):
    original_wp = np.array(wps, dtype=float)
    expanded = [original_wp[0, 0:3].copy()]
    gates = []
    n = original_wp.shape[0]
    for i in range(1, n - 1):
        center = original_wp[i, 0:3].copy()
        yaw_rad = math.radians(float(original_wp[i, 3]))
        forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
        pre = center - GATE_PRE * forward
        post = center + GATE_POST * forward
        expanded.extend([pre, center, post])
        gates.append({"center": center, "forward": forward, "radius": 0.5})
    expanded.append(original_wp[-1, 0:3].copy())
    return np.array(expanded, dtype=float), gates


def precompute_lengths(wp):
    seg_lens = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    return seg_lens, cum_s, float(cum_s[-1])


def point_at_s(s_target, wp, cum_s, total_len, seg_lens):
    s = np.clip(s_target, 0.0, total_len)
    if s <= 0.0:
        return wp[0].copy()
    if s >= total_len:
        return wp[-1].copy()
    idx = int(np.searchsorted(cum_s, s, side="right") - 1)
    idx = np.clip(idx, 0, len(seg_lens) - 1)
    ratio = (s - cum_s[idx]) / max(1e-9, seg_lens[idx])
    ratio = np.clip(ratio, 0.0, 1.0)
    return wp[idx] + ratio * (wp[idx + 1] - wp[idx])


def compute_projection(position, wp, cum_s, seg_lens, cur_seg_idx, sb=1, sf=2):
    start = max(0, cur_seg_idx - sb)
    end = min(len(seg_lens) - 1, cur_seg_idx + sf)
    best_d = float("inf")
    best_s = cum_s[cur_seg_idx]
    for i in range(start, end + 1):
        A, B = wp[i], wp[i + 1]
        v = B - A
        vv = float(np.dot(v, v))
        if vv < 1e-12:
            continue
        t = float(np.dot(position - A, v) / vv)
        t = np.clip(t, 0.0, 1.0)
        proj = A + t * v
        d = float(np.linalg.norm(position - proj))
        s_proj = float(cum_s[i] + t * seg_lens[i])
        if d < best_d:
            best_d = d
            best_s = s_proj
    return best_s, best_d


# ============================================================
# Gymnasium 环境
# ============================================================
@dataclass
class DroneRaceEnvConfig:
    waypoints: list = field(default_factory=lambda: DEFAULT_WAYPOINTS)
    max_steps: int = 1000              # ~33s @ 30Hz
    dt: float = 1.0 / 30.0
    action_scale: list = field(default_factory=lambda: [0.3, 0.3, 0.15])  # 每步最大位置增量 (m)
    lookahead_s: float = 0.5            # 固定前瞻距离 (m)
    pass_radius: float = 0.5
    start_noise_std: float = 0.05       # 起点位置噪声
    dynamics_noise_std: float = 0.01    # 执行噪声
    # 真实环境对接时置为 True: 每步通过 ROS 话题获取/发布真实状态
    use_ros_backend: bool = False


class DroneRaceEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "name": "DroneRace-v0"}

    def __init__(self, config: DroneRaceEnvConfig | None = None):
        super().__init__()
        self.cfg = config or DroneRaceEnvConfig()
        self.action_scale_arr = np.asarray(self.cfg.action_scale, dtype=float)
        self.wp, self.gates = build_expanded_path(self.cfg.waypoints)
        self.seg_lens, self.cum_s, self.total_len = precompute_lengths(self.wp)
        self.n_seg = len(self.seg_lens)

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # 内部状态
        self.pos = None
        self.vel = None
        self.prev_pos = None
        self.cur_seg_idx = 0
        self.s_progress = 0.0
        self.steps = 0
        self.pass_idx = 1
        self.error_sq_sum = 0.0
        self.error_count = 0
        self.gate_passed_flags = [False] * len(self.gates)
        self.start_time = None

    # ---------------------------------------------------------------
    # 核心接口
    # ---------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cfg = self.cfg  # keep
        start = self.wp[0].copy()
        noise = np.random.normal(0, self.cfg.start_noise_std, 3)
        noise[2] *= 0.3  # Z 噪声小一点
        self.pos = start + noise
        self.prev_pos = self.pos.copy()
        self.vel = np.zeros(3)
        self.cur_seg_idx = 0
        self.s_progress = 0.0
        self.steps = 0
        self.pass_idx = 1
        self.error_sq_sum = 0.0
        self.error_count = 0
        self.gate_passed_flags = [False] * len(self.gates)
        self.start_time = time.time()
        return self._get_obs().astype(np.float32), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(3)
        action = np.clip(action, -1.0, 1.0)

        # --- 1. 执行动作（简化动力学：目标→位置，含阻尼和噪声）---
        target_delta = action * self.action_scale_arr
        dynamics_noise = np.random.normal(0, self.cfg.dynamics_noise_std, 3)
        # 简化一阶惯性: 0.7*new + 0.3*保持
        desired = self.pos + target_delta
        self.pos = 0.7 * desired + 0.3 * self.pos + dynamics_noise
        self.pos[2] = np.clip(self.pos[2], 0.05, 5.0)  # 不钻地
        self.vel = (self.pos - self.prev_pos) / max(1e-6, self.cfg.dt)
        self.prev_pos = self.pos.copy()

        # --- 2. 几何进度计算 ---
        prev_s = self.s_progress
        s_new, terr = compute_projection(
            self.pos, self.wp, self.cum_s, self.seg_lens, self.cur_seg_idx
        )
        s_used = max(s_new, prev_s)
        s_used = min(s_used, self.total_len)
        self.s_progress = s_used
        s_delta = s_used - prev_s
        idx = int(np.searchsorted(self.cum_s, s_used, side="right") - 1)
        self.cur_seg_idx = np.clip(idx, 0, self.n_seg - 1)

        # 统计误差 (RMS 计算，和评分一致)
        self.error_sq_sum += terr * terr
        self.error_count += 1

        # --- 3. 过门检测 (顺序) ---
        if self.pass_idx < len(self.wp):
            wp_next = self.wp[self.pass_idx]
            d = float(np.linalg.norm(self.pos - wp_next))
            if d < self.cfg.pass_radius:
                self.pass_idx += 1
        # 真实门穿越 (3个中间门)
        for gi, g in enumerate(self.gates):
            if not self.gate_passed_flags[gi]:
                d_to_center = float(np.dot(self.pos - g["center"], g["forward"]))
                if abs(d_to_center) < 0.15:
                    radial = float(np.linalg.norm(
                        (self.pos - g["center"])
                        - d_to_center * g["forward"]
                    ))
                    if radial <= g["radius"]:
                        self.gate_passed_flags[gi] = True

        # --- 4. 终止条件 ---
        truncated = self.steps >= self.cfg.max_steps - 1
        finished_all = self.pass_idx >= len(self.wp)
        progress_ok = self.s_progress >= 0.99 * self.total_len
        terminated = finished_all and progress_ok
        all_gates = all(self.gate_passed_flags)

        # --- 5. Reward 设计 (和评分系统对齐!) ---
        rms = math.sqrt(self.error_sq_sum / max(1, self.error_count))
        flight_time = self.steps * self.cfg.dt

        # 精度奖励分量 (每一步 0~1，映射到 RMS)
        acc_component = 1.0 - np.clip(terr / 0.5, 0.0, 1.0)  # err<0时满, >0.5时0
        # 速度奖励分量: 进度增量/总长度 × scale
        speed_component = 10.0 * (s_delta / max(1e-6, self.total_len))
        # 完成奖励
        completion_bonus = 50.0 if terminated and all_gates else 0.0
        # 惩罚: 大误差
        error_penalty = -5.0 if terr > 1.0 else 0.0

        reward = float(0.5 * acc_component + 0.5 * speed_component + completion_bonus + error_penalty)

        # 结束时按总分给一个 final bonus
        if terminated or truncated:
            span_e = 0.50
            span_t = 50.0
            acc_s = 100.0 * (0.50 - rms) / span_e
            t_s = 100.0 * (60.0 - flight_time) / span_t
            acc_s = np.clip(acc_s, 0, 100)
            t_s = np.clip(t_s, 0, 100)
            final_sc = 0.5 * acc_s + 0.5 * t_s
            valid_bonus = 100.0 if all_gates else 0.0
            reward += float(final_sc * 0.5 + valid_bonus)

        self.steps += 1
        info = self._get_info()
        info["rms_error_m"] = round(rms, 5)
        info["flight_time_s"] = round(flight_time, 3)
        info["track_error_m"] = round(terr, 5)
        info["gates_passed"] = f"{sum(self.gate_passed_flags)}/{len(self.gates)}"
        info["run_valid"] = all_gates
        if terminated or truncated:
            info["final_score_predicted"] = round(
                0.5 * float(np.clip(100*(0.5-rms)/0.5,0,100))
                + 0.5 * float(np.clip(100*(60-flight_time)/50,0,100)), 2)

        return (
            self._get_obs().astype(np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        info = self._get_info()
        return (
            f"[Step {self.steps:>4}] "
            f"pos=({self.pos[0]:+.2f},{self.pos[1]:+.2f},{self.pos[2]:+.2f}) "
            f"prog={self.s_progress/self.total_len*100:5.1f}% "
            f"err={info.get('track_error_m',0):.3f}m "
            f"gates={info.get('gates_passed','?')}"
        )

    # ---------------------------------------------------------------
    # 内部
    # ---------------------------------------------------------------
    def _get_obs(self):
        proj = point_at_s(self.s_progress, self.wp, self.cum_s, self.total_len, self.seg_lens)
        la = point_at_s(self.s_progress + self.cfg.lookahead_s,
                        self.wp, self.cum_s, self.total_len, self.seg_lens)
        # 下一个门
        next_gate_center = None
        for gi, passed in enumerate(self.gate_passed_flags):
            if not passed:
                next_gate_center = self.gates[gi]["center"]
                break
        if next_gate_center is None:
            next_gate_center = self.wp[-1]

        obs = np.concatenate([
            self.pos,                                # 0:3
            self.vel,                                # 3:6
            [self.s_progress / self.total_len],      # 6
            [np.linalg.norm(self.pos - proj)],       # 7
            la - self.pos,                           # 8:11
            next_gate_center - self.pos,             # 11:14
        ]).astype(np.float32)
        # normalize large values
        obs[0:3] /= 5.0
        obs[3:6] /= 3.0
        obs[8:14] /= 5.0
        return np.clip(obs, -10.0, 10.0)

    def _get_info(self):
        return {
            "steps": self.steps,
            "s_progress_m": round(self.s_progress, 3),
            "s_progress_pct": round(self.s_progress / self.total_len * 100, 2),
            "waypoint_pass_idx": f"{self.pass_idx}/{len(self.wp)}",
        }


# ============================================================
# 接入 Stable-Baselines3 的示例 (需要有 SB3)
# ============================================================
def train_with_sb3(total_timesteps=200_000, save_path="drone_ppo.zip"):
    """
    需要先: pip install stable-baselines3
    NOTE: 两天内不建议跑，这是留作后续扩展
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("❌ 需要 stable-baselines3: pip install stable-baselines3")
        return
    env = DroneRaceEnv(DroneRaceEnvConfig(max_steps=800))
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"✅ 已保存模型: {save_path}")


def smoke_test(n_episodes=5):
    """不需要任何 RL 库，5分钟验证环境能跑通"""
    print(f"🚬 Smoke test: {n_episodes} 个随机回合\n")
    env = DroneRaceEnv(DroneRaceEnvConfig(max_steps=600))
    scores = []
    for ep in range(1, n_episodes + 1):
        obs, info = env.reset(seed=42 + ep)
        done = False
        total_r = 0.0
        while not done:
            action = env.action_space.sample()  # 随机动作
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            done = term or trunc
        final_score = info.get("final_score_predicted", float("nan"))
        valid = info.get("run_valid", False)
        scores.append(final_score)
        print(f"  Ep {ep:>2}: reward_sum={total_r:>7.2f}  "
              f"predict_score={final_score:.2f}  valid={valid}  "
              f"rms={info.get('rms_error_m','?')}m  time={info.get('flight_time_s','?')}s  "
              f"gates={info.get('gates_passed','?')}")
        if ep % 30 == 0:
            print(f"    ... {env.render()}")
    print(f"\n✅ 环境工作正常。{n_episodes} 回合平均预测分数 = {np.mean(scores):.2f}")
    print("   (随机策略分数当然低，这是预期的。训练 PPO/SAC 后会上来)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=5, help="smoke test 回合数")
    ap.add_argument("--train", action="store_true", help="用SB3训练 (需要stable-baselines3)")
    ap.add_argument("--train-steps", type=int, default=200_000)
    args = ap.parse_args()

    if args.train:
        train_with_sb3(total_timesteps=args.train_steps)
    else:
        smoke_test(args.smoke)
