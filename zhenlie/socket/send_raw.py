import socket
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# **服务器 IP（修改为你的主机 IP）**
SERVER_IP = "192.168.43.219"  # ⚠️ 你的实际笔记本 IP
SERVER_PORT = 5001  # **与 `receive_file.py` 保持一致**

# **监听的目录**
WATCH_DIR = "/home/mica/bin/record"  # ⚠️ 修改为实际路径

# **每次发送的块大小**
CHUNK_SIZE = 4096

# **存储已创建但未传输的文件夹**
pending_folders = []
last_folder_time = time.time()
FOLDER_WAIT_TIME = 31  # **等待31秒传输最后一个文件夹**


class FolderHandler(FileSystemEventHandler):
    """监听 `record` 目录，等待完整文件夹后传输"""

    def on_created(self, event):
        global last_folder_time
        if event.is_directory:
            folder_name = event.src_path
            print(f"📁 发现新文件夹: {folder_name}")

            if pending_folders:
                prev_folder = pending_folders.pop(0)
                print(f"📦 传输上一个文件夹: {prev_folder}")
                send_folder(prev_folder)

            pending_folders.append(folder_name)
            last_folder_time = time.time()


def send_folder(folder_path):
    """发送整个文件夹中的 `.raw` 文件"""
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".raw"):
            file_path = os.path.join(folder_path, file_name)
            send_file(file_path)


def send_file(file_path):
    """发送单个 `raw` 文件，并确保完整性"""
    try:
        file_size = os.path.getsize(file_path)  # 获取文件大小
        filename = os.path.basename(file_path)  # 获取文件名

        print(f"📡 发送 {filename}, 大小: {file_size} 字节")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((SERVER_IP, SERVER_PORT))  # 连接服务器
            print("✅ 连接成功，开始传输...")

            # **发送文件名长度 (4 字节) + 文件名**
            sock.sendall(len(filename).to_bytes(4, 'big'))  # 发送文件名长度
            sock.sendall(filename.encode('utf-8'))  # 发送文件名

            # **发送文件大小信息（8 字节）**
            sock.sendall(file_size.to_bytes(8, 'big'))  # 发送 8 字节文件大小

            # **开始发送文件内容**
            with open(file_path, "rb") as f:
                while True:
                    data = f.read(CHUNK_SIZE)
                    if not data:
                        break
                    sock.sendall(data)
                    time.sleep(0.001)  # **防止数据堆积，确保流畅传输**

            print(f"✅ 传输完成: {filename}")

    except Exception as e:
        print(f"❌ 发送失败: {e}")


if __name__ == "__main__":
    """启动文件夹监听（确保完整性）"""
    print(f"📡 正在监听目录: {WATCH_DIR}（包含所有子目录）")
    event_handler = FolderHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)

            if pending_folders and (time.time() - last_folder_time > FOLDER_WAIT_TIME):
                last_folder = pending_folders.pop(0)
                print(f"⏳ 超时，传输最后的文件夹: {last_folder}")
                send_folder(last_folder)

    except KeyboardInterrupt:
        observer.stop()
    observer.join()
