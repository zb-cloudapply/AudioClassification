import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyroomacoustics as pra

# 房间尺寸 (6m x 6m x 3m)
room_dim = [6, 6, 3]
mic_height = 1.5  # 麦克风高度设为 1.5m

# 采样率
fs = 16000


# 定义四种麦克风阵列结构（每个阵列都有8个阵元）
def create_mic_array(array_type, center=[3, 3], spacing=0.5):
    if array_type == "triangle":
        # 8个阵元，三角形阵列近似
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        positions = np.array([
            center[0] + spacing * np.cos(angles),
            center[1] + spacing * np.sin(angles)
        ])
    elif array_type == "rectangle":
        # 矩形阵列：8个阵元，矩形布局
        positions = np.array([
            [center[0] - spacing, center[1] - spacing],
            [center[0] + spacing, center[1] - spacing],
            [center[0] - spacing, center[1] + spacing],
            [center[0] + spacing, center[1] + spacing],
            [center[0] - spacing / 2, center[1] - spacing / 2],
            [center[0] + spacing / 2, center[1] - spacing / 2],
            [center[0] - spacing / 2, center[1] + spacing / 2],
            [center[0] + spacing / 2, center[1] + spacing / 2]
        ]).T
    elif array_type == "cross":
        # 十字形阵列：8个阵元，十字形布局
        positions = np.array([
            [center[0] - spacing, center[1]],
            [center[0] + spacing, center[1]],
            [center[0], center[1] - spacing],
            [center[0], center[1] + spacing],
            [center[0] - spacing / 2, center[1] - spacing / 2],
            [center[0] + spacing / 2, center[1] - spacing / 2],
            [center[0] - spacing / 2, center[1] + spacing / 2],
            [center[0] + spacing / 2, center[1] + spacing / 2]
        ]).T
    elif array_type == "circular":
        # 圆形阵列：8个阵元，均匀分布在圆形上
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        positions = np.array([
            center[0] + spacing * np.cos(angles),
            center[1] + spacing * np.sin(angles)
        ])
    else:
        raise ValueError("Unsupported array type")

    # 添加高度信息（所有麦克风在相同高度）
    positions = np.vstack((positions, np.full(positions.shape[1], mic_height)))
    return positions


# 创建房间，设置声源和麦克风阵列
def create_room_and_mic(array_type):
    # 创建房间环境
    room = pra.ShoeBox(room_dim, fs=fs, max_order=10)

    # 创建麦克风阵列
    mic_positions = create_mic_array(array_type)
    mic_array = pra.MicrophoneArray(mic_positions, fs)
    room.add_microphone_array(mic_array)

    # 声源位置
    source_position = [np.random.uniform(1, 5), np.random.uniform(1, 5), mic_height]

    # 添加声源
    room.add_source(source_position)

    # 计算房间脉冲响应
    room.compute_rir()

    return room, mic_positions, source_position


# 计算波束响应（3D）
def compute_beamforming_energy(room, mic_positions, source_position, grid_x, grid_y):
    # 计算每个点的能量
    energy_map = np.zeros((len(grid_y), len(grid_x)))

    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            distances = np.linalg.norm(mic_positions[:2].T - np.array([x, y]), axis=1)
            energy_map[j, i] = np.mean(np.exp(-distances))  # 能量与距离的反比

    return energy_map


# 计算主瓣宽度和副瓣比
def compute_beamwidth_and_sidelobe_ratio(energy_map, grid_x, grid_y):
    # 找到最大能量（即主瓣位置）
    max_energy = np.max(energy_map)
    main_lobe_indices = np.where(energy_map == max_energy)

    # 主瓣宽度：计算最大能量位置附近的半能量宽度（beamwidth）
    half_energy = max_energy / 2
    beamwidth_x = np.max(grid_x[main_lobe_indices[1]]) - np.min(grid_x[main_lobe_indices[1]])
    beamwidth_y = np.max(grid_y[main_lobe_indices[0]]) - np.min(grid_y[main_lobe_indices[0]])

    # 副瓣计算：副瓣最大值与主瓣值的比值
    sidelobe_energy = energy_map[energy_map < max_energy]
    sidelobe_max = np.max(sidelobe_energy)
    sidelobe_ratio = sidelobe_max / max_energy

    return beamwidth_x, beamwidth_y, sidelobe_ratio


# 计算声源定位误差
def compute_localization_error(true_position, estimated_position):
    return np.linalg.norm(np.array(true_position) - np.array(estimated_position))


# 计算信噪比
def compute_snr(mic_positions, source_position):
    distances = np.linalg.norm(mic_positions[:2].T - np.array(source_position[:2]), axis=1)
    signal_energy = np.mean(1 / distances ** 2)  # 假设信号强度与距离的平方成反比
    noise_energy = np.mean(1 / (distances + 1) ** 2)  # 假设噪声衰减模式
    snr = 10 * np.log10(signal_energy / noise_energy)
    return snr


# 主程序：遍历不同阵列类型，计算并比较性能
results = {}

grid_x = np.linspace(0, room_dim[0], 50)  # X轴的网格点
grid_y = np.linspace(0, room_dim[1], 50)  # Y轴的网格点

for array_type in ["triangle", "rectangle", "cross", "circular"]:
    print(f"Processing {array_type} array...")

    # 创建房间和麦克风阵列
    room, mic_positions, source_position = create_room_and_mic(array_type)

    # 计算波束响应能量
    energy_map = compute_beamforming_energy(room, mic_positions, source_position, grid_x, grid_y)

    # 计算主瓣宽度、副瓣比
    beamwidth_x, beamwidth_y, sidelobe_ratio = compute_beamwidth_and_sidelobe_ratio(energy_map, grid_x, grid_y)

    # 计算定位误差（此处使用简化方法：取麦克风阵列的重心作为估计位置）
    estimated_position = np.mean(mic_positions, axis=1)  # 估计位置
    localization_error = compute_localization_error(source_position, estimated_position)

    # 计算信噪比
    snr = compute_snr(mic_positions, source_position)

    # 存储结果
    results[array_type] = {
        "beamwidth_x": beamwidth_x,
        "beamwidth_y": beamwidth_y,
        "sidelobe_ratio": sidelobe_ratio,
        "localization_error": localization_error,
        "snr": snr
    }

    # 绘制波束图（3D）
    X, Y = np.meshgrid(grid_x, grid_y)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制波束响应能量图
    ax.plot_surface(X, Y, energy_map, cmap='inferno', edgecolor='none')
    ax.set_xlabel('X position [m]')
    ax.set_ylabel('Y position [m]')
    ax.set_zlabel('Energy')
    ax.set_title(f'{array_type.capitalize()} Array - Beamforming Energy Map')

    plt.show()

# 输出结果
for array_type, metrics in results.items():
    print(f"\n{array_type.capitalize()} Array:")
    print(f"  Beamwidth (X): {metrics['beamwidth_x']:.2f} m")
    print(f"  Beamwidth (Y): {metrics['beamwidth_y']:.2f} m")
    print(f"  Sidelobe Ratio: {metrics['sidelobe_ratio']:.2f}")
    print(f"  Localization Error: {metrics['localization_error']:.2f} m")
    print(f"  SNR: {metrics['snr']:.2f} dB")
