import numpy as np

def generate_distance_feature_vector(B_w, v_w_test, fc, T, C, Ns):
    
    # 1. 构建速度补偿向量 C_test，对应公式 (12) 
    # 使用 0 到 Ns-1 的索引
    n_indices = np.arange(Ns).reshape(Ns, 1)
    
    # 公式 (12) 中的指数项
    phase_C_test = -1j * 2 * np.pi * fc * (2 * v_w_test * n_indices * T) / C
    C_test = np.exp(phase_C_test)
    
    # 2. 计算 E_w = B_w @ C_test，对应公式 (13) 
    E_w = B_w @ C_test
    
    return E_w

def generate_velocity_feature_vector(B_w, R_w_test, delta_f, C, Nc):
    
    # 1. 构建距离补偿向量 A_test，对应公式 (14) 
    # 使用 0 到 Nc-1 的索引
    m_indices = np.arange(Nc).reshape(1, Nc)
    
    # 公式 (14) 中的指数项
    phase_A_test = 1j * 2 * np.pi * m_indices * delta_f * (2 * R_w_test) / C
    A_test = np.exp(phase_A_test)
    
    # 2. 计算 F_w = A_test @ B_w，对应公式 (15) 
    F_w = A_test @ B_w
    
    return F_w

if __name__ == "__main__":
    
    # --- 1. 定义系统参数 (来自论文) 
    fc = 24e9        # 载波频率 (Hz)
    B = 93.1e6       # 信号带宽 (Hz)
    T = 12.375e-6    # OFDM 符号周期 (s)
    Nc = 128         # 子载波数量
    Ns = 256         # OFDM 符号数量
    C = 3e8          # 光速 (m/s)
    delta_f = 1 / (T) 
    
    # --- 2. 模拟输入数据 ---
        # 假设的真实目标参数
    R_w_true = 150.0  # 真实距离 (m)
    v_w_true = 20.0   # 真实径向速度 (m/s)
    
    # 假设的估计值 (来自 III-A.2 的2D FFT步骤) 
    # 估计值通常有微小误差
    R_w_test = 150.1  # 估计的距离 (m)
    v_w_test = 20.05  # 估计的径向速度 (m/s)
    
    # 假设的信道增益
    U_w = 0.8 * np.exp(1j * np.pi / 4) # 任意复数增益
    
    # 模拟构建基础矩阵 B_w (Nc, Ns)，对应公式 (6) 
    # B_w[m, n] = U_w * exp(j*velo_phase) * exp(j*dist_phase)
    
    m_indices = np.arange(Nc).reshape(Nc, 1) # (Nc, 1)
    n_indices = np.arange(Ns).reshape(1, Ns) # (1, Ns)
    
    # 距离相关的相位 (Nc, 1)
    dist_phase = np.exp(-1j * 2 * np.pi * m_indices * delta_f * (2 * R_w_true / C))
    
    # 速度相关的相位 (1, Ns)
    velo_phase = np.exp(1j * 2 * np.pi * fc * (2 * v_w_true * n_indices * T / C))
    
    # 使用numpy的广播特性通过外积组合相位
    B_w_signal = U_w * (dist_phase @ velo_phase)
    
    # 模拟加性高斯白噪声
    SNR_dB = -5 # 假设SNR为-5dB
    signal_power = np.mean(np.abs(B_w_signal)**2)
    noise_power = signal_power / (10**(SNR_dB / 10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(Nc, Ns) + 1j * np.random.randn(Nc, Ns))
    
    B_w = B_w_signal + noise
    
    print(f"--- 输入参数 ---")
    print(f"基础矩阵 B_w 形状: {B_w.shape} (Nc, Ns)")
    print(f"估计距离 R_w_test: {R_w_test:.2f} m")
    print(f"估计速度 v_w_test: {v_w_test:.2f} m/s")
    print("-" * 20)
    
    # --- 3. 生成特征向量 ---
    
    # 生成距离特征向量 E_w
    E_w = generate_distance_feature_vector(B_w, v_w_test, fc, T, C, Ns)
    
    # 生成速度特征向量 F_w
    F_w = generate_velocity_feature_vector(B_w, R_w_test, delta_f, C, Nc)
    
    print(f"--- 输出结果 ---")
    print(f"距离特征向量 E_w 形状: {E_w.shape} (Nc, 1)")
    print(f"速度特征向量 F_w 形状: {F_w.shape} (1, Ns)")
