import numpy as np
import wave

# 你的 raw 文件路径
raw_file = r"G:\dataset\zhenlie\record\2025-03-06_20-57-59\audio_opt3.raw"
# raw_file = r"D:\Learning\192.168.3.102\202503061449\audio_opt3.raw"

wav_file = "output.wav"

# 先检查 raw 数据
with open(raw_file, "rb") as f:
    raw_data = f.read()

# 尝试 int16 和 float32 解析
try:
    data_int16 = np.frombuffer(raw_data, dtype=np.int16)
    print(f"int16 数据范围: {np.min(data_int16)} ~ {np.max(data_int16)}")
except:
    print("int16 解析失败")

try:
    data_float32 = np.frombuffer(raw_data, dtype=np.float32)
    print(f"float32 数据范围: {np.min(data_float32)} ~ {np.max(data_float32)}")
except:
    print("float32 解析失败")

# 选择正确的格式
raw_data = np.frombuffer(raw_data, dtype=np.int16)  # 如果 float32 解析正确，改成 np.float32

# 假设是单通道
channels = 1  # 先用单通道
sample_width = 2  # 16-bit PCM
sample_rate = 48000  # 可能是 16000、32000、48000，尝试不同值

# 计算音频时长
duration = len(raw_data) / (channels * sample_rate)
print(f"计算音频时长: {duration:.2f} 秒")

# 如果音量太小，放大
raw_data = raw_data * 100  # 放大 10 倍

# 保存 WAV
with wave.open(wav_file, "w") as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(sample_rate)
    wf.writeframes(raw_data.tobytes())

print(f"WAV 文件已保存：{wav_file}")
