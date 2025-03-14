import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib

# 设置 Matplotlib 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 参数设定
sample_rate = 48000  # 采样率
speed_of_sound = 343.2  # 声速 (m/s)
mic_radius = 0.0425  # 阵列半径（单位：米）
num_mics = 8  # 环形阵列 8 麦克风

# 读取 8 个 raw 文件
raw_files = [
    r"G:\dataset\zhenlie\audio_opt0.raw",
    r"G:\dataset\zhenlie\audio_opt1.raw",
    r"G:\dataset\zhenlie\audio_opt2.raw",
    r"G:\dataset\zhenlie\audio_opt3.raw",
    r"G:\dataset\zhenlie\audio_opt4.raw",
    r"G:\dataset\zhenlie\audio_opt5.raw",
    r"G:\dataset\zhenlie\audio_opt6.raw",
    r"G:\dataset\zhenlie\audio_opt7.raw",
]

# 读取数据
raw_data_list = []
for file in raw_files:
    with open(file, "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=np.int16)
        raw_data_list.append(raw_data)

# 确保所有通道长度一致
min_length = min(len(data) for data in raw_data_list)
raw_data_list = [data[:min_length] for data in raw_data_list]

# 组合成矩阵 (samples, 8)
multi_channel_data = np.stack(raw_data_list, axis=1)

# 使用 GCC-PHAT 计算 TDoA
def gcc_phat(signal1, signal2, sample_rate):
    """使用 GCC-PHAT 计算 TDoA"""
    n = len(signal1) + len(signal2)
    fft1 = np.fft.rfft(signal1, n=n)
    fft2 = np.fft.rfft(signal2, n=n)
    cross_spectrum = fft1 * np.conj(fft2)
    cross_spectrum /= np.abs(cross_spectrum)  # PHAT 归一化
    corr = np.fft.irfft(cross_spectrum, n=n)
    max_lag = len(signal1) - 1
    lag_index = np.argmax(np.abs(corr)) - max_lag
    tdoa = lag_index / sample_rate
    return tdoa, lag_index

# 计算 TDoA（相对于 Mic 0）
tdoa_list = []
for i in range(1, num_mics):  # 只计算相对于 `Mic 0`
    tdoa, lag = gcc_phat(multi_channel_data[:, 0], multi_channel_data[:, i], sample_rate)
    tdoa_list.append((0, i, tdoa))
    print(f"TDoA between Mic 0 and Mic {i}: {tdoa:.6f} s (Lag: {lag})")

# 计算 DOA 角度
mic_angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
source_positions = []

for i, j, tdoa in tdoa_list:
    distance_diff = np.clip(tdoa * speed_of_sound, -mic_radius, mic_radius)  # 限制范围
    angle = np.arcsin(distance_diff / mic_radius)  # 计算角度
    angle_deg = np.degrees(angle)  # 角度转换

    # 计算声源坐标
    x = mic_radius * np.cos(mic_angles[i]) + distance_diff * np.cos(angle)
    y = mic_radius * np.sin(mic_angles[i]) + distance_diff * np.sin(angle)
    source_positions.append((x, y))

# 计算唯一声源位置（去掉异常值后求均值）
source_positions = np.array(source_positions)
valid_positions = source_positions[np.abs(source_positions[:, 0]) < 2 * mic_radius]
avg_x = np.mean(valid_positions[:, 0])
avg_y = np.mean(valid_positions[:, 1])

print(f"最终声源估计坐标: ({avg_x:.3f}, {avg_y:.3f})")

# 绘图
plt.figure(figsize=(8, 8))

# 画出麦克风位置
mic_x = mic_radius * np.cos(mic_angles)
mic_y = mic_radius * np.sin(mic_angles)
plt.scatter(mic_x, mic_y, c='b', label="麦克风")

# 画出最终声源坐标
plt.scatter(avg_x, avg_y, c='r', s=100, label="最终估计声源")

plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlim(-mic_radius * 2, mic_radius * 2)
plt.ylim(-mic_radius * 2, mic_radius * 2)
plt.legend()
plt.title("声源定位坐标")
plt.xlabel("X 轴 (m)")
plt.ylabel("Y 轴 (m)")
plt.grid()
plt.show()
