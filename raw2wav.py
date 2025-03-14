import numpy as np
import wave
import os
import time

def convert_raw_to_wav(folder_path, wav_output_folder, sample_rate=48000, volume_boost=10, timeout=30):
    """ 直接处理 `folder_path`，将其中的 8 个 RAW 文件合并为 WAV """

    print(f"📂 处理文件夹: {folder_path}")

    # **等待 `raw` 文件完整写入**
    required_files = [f"audio_opt{i}.raw" for i in range(8)]
    start_time = time.time()

    while time.time() - start_time < timeout:
        existing_files = [f for f in required_files if os.path.exists(os.path.join(folder_path, f))]
        if len(existing_files) == 8:
            print(f"✅ 所有 raw 文件已写入")
            break
        print(f"⌛ 等待 raw 文件完整写入... 已检测到 {len(existing_files)}/8 个文件")
        time.sleep(1)

    raw_files = [os.path.join(folder_path, f) for f in required_files]
    missing_files = [f for f in raw_files if not os.path.exists(f)]

    if missing_files:
        raise FileNotFoundError(f"❌ 缺少以下音频文件: {missing_files}")

    # **生成 WAV 文件名**
    folder_name = os.path.basename(folder_path)
    wav_filename = f"{folder_name}.wav"
    wav_output_path = os.path.join(wav_output_folder, wav_filename)

    raw_data_list = []
    max_length = 0  # 记录最长通道长度

    for file in raw_files:
        with open(file, "rb") as f:
            raw_bytes = f.read()

            if len(raw_bytes) == 0:
                print(f"⚠️ 文件 {file} 为空，跳过处理")
                continue

            if len(raw_bytes) % 2 != 0:
                print(f"⚠️ 文件 {file} 长度异常，去掉最后 1 字节以对齐 int16 格式")
                raw_bytes = raw_bytes[:-1]

            raw_data = np.frombuffer(raw_bytes, dtype=np.int16)

            # **记录最长通道长度**
            if len(raw_data) > max_length:
                max_length = len(raw_data)

            raw_data_list.append(raw_data)

    if len(raw_data_list) < 8:
        raise ValueError(f"❌ {folder_path} 的数据不完整，只有 {len(raw_data_list)} 个通道数据")

    # **对所有通道进行补零填充，保证时长一致**
    for i in range(len(raw_data_list)):
        if len(raw_data_list[i]) < max_length:
            raw_data_list[i] = np.pad(raw_data_list[i], (0, max_length - len(raw_data_list[i])), 'constant')

    # **拼接成 8 通道数据**
    multi_channel_data = np.column_stack(raw_data_list)

    # **音量放大**
    multi_channel_data = multi_channel_data * volume_boost
    multi_channel_data = np.clip(multi_channel_data, -32768, 32767)

    # **保存 WAV**
    with wave.open(wav_output_path, "w") as wf:
        wf.setnchannels(8)  # 8 通道
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)  # 采样率
        wf.writeframes(multi_channel_data.astype(np.int16).tobytes())

    print(f"✅ 多通道 WAV 文件已保存：{wav_output_path}")
    return wav_output_path
