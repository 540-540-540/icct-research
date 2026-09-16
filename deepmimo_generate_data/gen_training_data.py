"""
V22 数据生成: 含 FFT 处理增益 + 假目标的物理修正信号模型

轨迹类型:
  Type A (60%): 直线段 + 圆弧段拼接 (城市道路)
  Type C (40%): S 形曲线 / 蜿蜒道路

信号模型修正 (V22 新增):
  ★ FFT 处理增益: 距离谱 +21dB (128点), 速度谱 +24dB (256点)
    - 0dB 输入 → 后处理 ~21dB: 峰值清晰, 但有假目标干扰
    - 20dB 输入 → 后处理 ~41dB: 几乎无噪声
  ★ 假目标生成: 杂波旁瓣/多径干扰
    - 低 SNR 下更频繁、幅度更强, 偶尔接近真峰值
"""
import numpy as np
import os, argparse
from scipy.constants import c as C

# ─── OFDM 参数 ───
fc = 28e9; df = 720e3; Nc = 128; Ns = 256
Tsym = 1 / df
range_res = C / (2 * Nc * df)
vel_res = (C / (2 * fc)) / (Ns * Tsym)

BS_POS = np.array([[0, 0], [150, 0], [75, 150]], dtype=np.float64)


# ═══════════════════════════════════════════════════════════
# 轨迹生成器
# ═══════════════════════════════════════════════════════════

def gen_type_a(T, dt, rng):
    """Type A: 直线段 + 圆弧段拼接 (城市道路)
    变体: 转弯角度、段数、速度变化幅度随机化"""
    x, y = rng.uniform(15, 135), rng.uniform(15, 135)
    heading = rng.uniform(0, 2 * np.pi)
    speed = rng.uniform(5, 12)
    locs = np.zeros((T, 2))

    # 随机 3-8 个路段
    n_segments = rng.randint(3, 9)
    seg_lengths = rng.dirichlet(np.ones(n_segments)) * T
    seg_types = rng.choice(['straight', 'gentle_turn', 'sharp_turn'], n_segments,
                            p=[0.4, 0.35, 0.25])
    seg_idx = 0; seg_progress = 0
    turn_dir = 0.0

    for t in range(T):
        locs[t] = [x, y]
        if seg_idx < n_segments:
            seg_type = seg_types[seg_idx]
            if seg_type == 'straight':
                turn_dir = 0.0
                # 直道上随机加减速
                speed += rng.uniform(-0.1, 0.15)
            elif seg_type == 'gentle_turn':
                if seg_progress == 0:
                    turn_dir = rng.choice([-1, 1]) * rng.uniform(0.15, 0.4)
                speed = max(speed * 0.998, 4)  # 轻微减速
            else:  # sharp_turn
                if seg_progress == 0:
                    turn_dir = rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
                speed = max(speed * 0.995, 3)  # 明显减速

            heading += turn_dir * dt
            speed = np.clip(speed, 3, 15)

            seg_progress += 1
            if seg_progress >= seg_lengths[seg_idx]:
                seg_idx += 1; seg_progress = 0
                # 段间速度恢复
                speed = np.clip(speed + rng.uniform(0, 2), 4, 14)

        # 微小随机扰动 (路面/风)
        heading += 0.008 * rng.randn()
        speed += 0.02 * rng.randn()
        speed = np.clip(speed, 2, 16)
        x += speed * np.cos(heading) * dt
        y += speed * np.sin(heading) * dt

        # 软边界: 接近边界时转向
        margin = 8
        if x < margin:   heading = abs(heading) if abs(heading) > np.pi/2 else rng.uniform(0.2, 1.0)
        if x > 150-margin: heading = np.pi - heading if heading > 0 else -(np.pi + heading)
        if y < margin:   heading = abs(heading)
        if y > 150-margin: heading = -abs(heading)
        x = np.clip(x, 2, 148)
        y = np.clip(y, 2, 148)

    return locs


def gen_type_c(T, dt, rng):
    """Type C: S 形曲线 / 蜿蜒道路
    变体: 频率、振幅、基础方向、速度变化随机化"""
    x = rng.uniform(10, 140)
    y = rng.uniform(10, 140)
    speed = rng.uniform(5, 12)
    heading = rng.uniform(0, 2 * np.pi)  # 任意初始方向

    # S 曲线参数 (多样化)
    freq = rng.uniform(0.004, 0.02)
    amp = rng.uniform(0.2, 0.7)
    # 可选: 叠加第二个频率 (更复杂的弯道)
    freq2 = rng.uniform(0.001, 0.008)
    amp2 = rng.uniform(0.0, 0.3)
    phase = rng.uniform(0, 2 * np.pi)

    locs = np.zeros((T, 2))
    for t in range(T):
        locs[t] = [x, y]
        # 正弦调制转向 (双频)
        turn_rate = (amp * np.sin(2 * np.pi * freq * t + phase) +
                     amp2 * np.sin(2 * np.pi * freq2 * t))
        heading += turn_rate * dt
        # 速度随弯道变化: 转弯时减速
        speed_target = rng.uniform(8, 13) - abs(turn_rate) * 3
        speed += 0.1 * (speed_target - speed)
        speed = np.clip(speed + 0.03 * rng.randn(), 3, 15)
        x += speed * np.cos(heading) * dt
        y += speed * np.sin(heading) * dt

        # 软边界
        margin = 8
        if x < margin:   heading = abs(heading) if abs(heading) > np.pi/2 else rng.uniform(0.2, 1.0)
        if x > 150-margin: heading = np.pi - heading if heading > 0 else -(np.pi + heading)
        if y < margin:   heading = abs(heading)
        if y > 150-margin: heading = -abs(heading)
        x = np.clip(x, 2, 148)
        y = np.clip(y, 2, 148)

    return locs


def gen_vehicle_trajectory(T=300, dt=0.05, rng=None):
    """60% 城市道路 + 40% S 曲线"""
    if rng is None: rng = np.random.RandomState()
    if rng.rand() < 0.6:
        return gen_type_a(T, dt, rng)
    else:
        return gen_type_c(T, dt, rng)


# ═══════════════════════════════════════════════════════════
# OFDM 感知信号生成
# ═══════════════════════════════════════════════════════════

def gen_ofdm_sensing(true_dist, true_vel, snr_db, rng=None):
    """
    生成 OFDM 感知频谱 (含 FFT 处理增益 + 假目标)

    物理模型:
      1. FFT 处理增益: N 点 FFT 在频域提供 10*log10(N) dB 的增益
         - 距离谱 128 点: +21.1 dB
         - 速度谱 256 点: +24.1 dB
      2. 后处理 SNR = 输入 SNR + 处理增益
         - 0dB 输入 → 距离谱后处理 ~21dB (峰值清晰, 但偶有假目标)
         - 20dB 输入 → 距离谱后处理 ~41dB (干净)
      3. 假目标 (杂波旁瓣 / 多径干扰):
         - 随机出现在频谱中, 低 SNR 下更多更强
         - 偶尔幅度接近真实峰值, 导致 argmax 判错
    """
    if rng is None: rng = np.random.RandomState()
    T = len(true_dist)
    snr_linear = 10 ** (snr_db / 10)

    # ★ FFT 处理增益
    dist_pg = Nc   # 128 点 FFT → 增益 ~21 dB
    vel_pg  = Ns   # 256 点 FFT → 增益 ~24 dB

    # 后处理有效 SNR (线性域)
    eff_dist_snr = snr_linear * dist_pg
    eff_vel_snr  = snr_linear * vel_pg

    dist_spectra = np.zeros((T, Nc), dtype=np.float32)
    vel_spectra  = np.zeros((T, Ns), dtype=np.float32)

    # 假目标参数: 低 SNR → 更多、更强
    false_rate = np.clip(2.5 - snr_db / 10.0, 0.3, 5.0)
    # 假目标幅度 (相对信号峰值=1.0):
    #   正 SNR: 保持原公式 (不影响已有 0/5/10dB 数据)
    #   负 SNR: 线性映射, 平缓渐进
    #     -5dB  → scale=1.20 → max幅=1.08 (argmax错误率~20%)
    #    -10dB  → scale=1.40 → max幅=1.26 (argmax错误率~40%)
    if snr_db >= 0:
        false_amp_scale = min(1.0 / np.sqrt(snr_linear + 0.01), 4.0)
    else:
        false_amp_scale = 1.0 + (-snr_db) * 0.04

    for t in range(T):
        # ═══ 距离谱 ═══
        d_bin = true_dist[t] / range_res
        d_bin_idx = int(np.clip(d_bin, 0, Nc - 1))

        # 热噪声底 (含处理增益, 大幅压低)
        d_spec = rng.rayleigh(1.0 / np.sqrt(eff_dist_snr), size=Nc)

        # 真实目标峰值 (高斯形, 峰值≈1.0)
        for k in range(max(0, d_bin_idx - 3), min(Nc, d_bin_idx + 4)):
            d_spec[k] += np.exp(-0.5 * ((k - d_bin) / 0.8) ** 2)

        # ★ 假目标 (杂波旁瓣 / 多径)
        n_false_d = rng.poisson(false_rate)
        for _ in range(n_false_d):
            fb = rng.randint(0, Nc)
            fa = rng.uniform(0.3, 0.9) * false_amp_scale
            # 假目标也有宽度 (不是单 bin 尖刺)
            fw = rng.uniform(0.5, 1.5)
            for k in range(max(0, fb - 2), min(Nc, fb + 3)):
                d_spec[k] += fa * np.exp(-0.5 * ((k - fb) / fw) ** 2)

        dist_spectra[t] = d_spec.astype(np.float32)

        # ═══ 速度谱 ═══
        v_bin = true_vel[t] / vel_res + Ns / 2
        v_bin_idx = int(np.clip(v_bin, 0, Ns - 1))

        v_spec = rng.rayleigh(1.0 / np.sqrt(eff_vel_snr), size=Ns)

        for k in range(max(0, v_bin_idx - 3), min(Ns, v_bin_idx + 4)):
            v_spec[k] += np.exp(-0.5 * ((k - v_bin) / 0.8) ** 2)

        # ★ 假目标
        n_false_v = rng.poisson(false_rate)
        for _ in range(n_false_v):
            fb = rng.randint(0, Ns)
            fa = rng.uniform(0.3, 0.9) * false_amp_scale
            fw = rng.uniform(0.5, 1.5)
            for k in range(max(0, fb - 2), min(Ns, fb + 3)):
                v_spec[k] += fa * np.exp(-0.5 * ((k - fb) / fw) ** 2)

        vel_spectra[t] = v_spec.astype(np.float32)

    return dist_spectra, vel_spectra


def gen_one_file(file_idx, n_traj, T, snr_db, out_dir, rng):
    all_locs = np.zeros((n_traj, T, 2), dtype=np.float32)
    all_dist = [np.zeros((n_traj, T, Nc), dtype=np.float16) for _ in range(3)]
    all_vel = [np.zeros((n_traj, T, Ns), dtype=np.float16) for _ in range(3)]
    all_mask = np.ones((n_traj, T, 3), dtype=np.float32)

    for j in range(n_traj):
        locs = gen_vehicle_trajectory(T=T, rng=rng)
        all_locs[j] = locs

        for b in range(3):
            dx = locs[:, 0] - BS_POS[b, 0]
            dy = locs[:, 1] - BS_POS[b, 1]
            true_dist = np.sqrt(dx**2 + dy**2)
            true_vel = (dx[:-1] * np.diff(locs[:, 0]) / 0.05 +
                        dy[:-1] * np.diff(locs[:, 1]) / 0.05) / (true_dist[:-1] + 1e-9)
            true_vel = np.concatenate([[true_vel[0]], true_vel])

            d_spec, v_spec = gen_ofdm_sensing(true_dist, true_vel, snr_db, rng)
            all_dist[b][j] = d_spec.astype(np.float16)
            all_vel[b][j] = v_spec.astype(np.float16)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'gt_locs_{file_idx}.npy'), all_locs)
    np.save(os.path.join(out_dir, f'bs_mask_{file_idx}.npy'), all_mask)
    for b in range(3):
        np.save(os.path.join(out_dir, f'bs{b+1}_dist_{file_idx}.npy'), all_dist[b])
        np.save(os.path.join(out_dir, f'bs{b+1}_vel_{file_idx}.npy'), all_vel[b])
    print(f"  File {file_idx}: {n_traj} trajs, SNR={snr_db}dB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./dataset_final_npy')
    parser.add_argument('--file_num', type=int, default=20)
    parser.add_argument('--n_traj', type=int, default=64)
    parser.add_argument('--T', type=int, default=300)
    parser.add_argument('--snr_list', type=str, default='-10,-5,0,5,10')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    snr_list = [int(s) for s in args.snr_list.split(',')]
    rng = np.random.RandomState(args.seed)

    for snr in snr_list:
        out_dir = os.path.join(args.output_dir, f'{snr}dB')
        print(f"\n=== Generating SNR={snr}dB, {args.file_num} files ===")
        for fi in range(args.file_num):
            gen_one_file(fi, args.n_traj, args.T, snr, out_dir, rng)

    print(f"\nDone! Generated data for SNR={snr_list}")
    print(f"Trajectory types: 60% city-road (straight+turns), 40% S-curve")


if __name__ == '__main__':
    main()
