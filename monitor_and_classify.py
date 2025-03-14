import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from raw2wav import convert_raw_to_wav
from infer import wav_classify

class SoundMonitor(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            return  # 只监听文件夹

        folder_path = event.src_path  # 新增的文件夹路径
        print(f"📂 检测到新文件夹: {folder_path}")

        # **等待 `raw` 文件完全写入**
        self.wait_for_raw_files(folder_path)

        predict_dir = convert_raw_to_wav(folder_path, outwave_path, volume_boost=100)
        print(f"✅ 转换完成: {predict_dir}")

        # **进行分类**
        wav_classify(predict_dir)
        print(f"🎵 分类完成: {predict_dir}")

    def wait_for_raw_files(self, folder_path, required_files=8, timeout=30):
        """
        等待 `folder_path` 下 `raw` 文件完整写入
        - `required_files=8` : 需要 `audio_opt0.raw` ~ `audio_opt7.raw` 共 8 个文件
        - `timeout=30` : 最长等待 30 秒
        """
        print(f"⌛ 等待 {folder_path} 下的 raw 文件写入...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            raw_files = [f"audio_opt{i}.raw" for i in range(required_files)]
            existing_files = [f for f in raw_files if os.path.exists(os.path.join(folder_path, f))]

            if len(existing_files) == required_files:
                print(f"✅ 所有 {required_files} 个 raw 文件已写入: {existing_files}")
                return

            time.sleep(1)  # 休眠 1 秒后继续检查

        print(f"⚠️ 超时等待 {timeout} 秒，仍然未检测到所有 raw 文件，可能缺失部分数据。")

if __name__ == "__main__":
    record_path = r"G:\dataset\zhenlie\record"
    outwave_path = r"G:\dataset\zhenlie\output"

    event_handler = SoundMonitor()
    observer = Observer()
    observer.schedule(event_handler, path=record_path, recursive=True)
    observer.start()

    try:
        print(f"🔍 监听 {record_path} 目录，等待新数据...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
