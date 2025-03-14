import socket
import os
import datetime

# 服务器监听端口
SERVER_PORT = 5001
SAVE_DIR = r"G:\dataset\zhenlie\record"  # 确保路径正确

# 计数变量，每 10 个文件创建一个新文件夹
file_count = 0
current_folder = None  # 记录当前的保存文件夹

def get_new_save_folder():
    """每 10 个文件创建一个新的时间戳文件夹"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(SAVE_DIR, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def receive_file():
    """ 服务器监听并接收 `raw` 文件 """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.bind(("0.0.0.0", SERVER_PORT))  # 监听所有 IP
        server_sock.listen(5)
        print(f"📡 服务器启动，等待开发板连接 {SERVER_PORT} ...")

        file_count = 0  # 计数，每 10 个文件创建新文件夹
        current_folder = get_new_save_folder()

        while True:
            conn, addr = server_sock.accept()
            print(f"✅ 连接成功: {addr}")

            try:
                # **先接收文件名长度 (4 字节)**
                filename_length = int.from_bytes(conn.recv(4), 'big')
                filename = conn.recv(filename_length).decode('utf-8')  # 接收文件名

                # **再接收 8 字节文件大小**
                file_size = int.from_bytes(conn.recv(8), 'big')
                print(f"📦 接收文件: {filename}, 预计大小: {file_size} 字节")

                # **检查文件夹是否需要更换**
                if file_count % 10 == 0:
                    current_folder = get_new_save_folder()
                file_count += 1

                # **动态生成保存路径**
                save_path = os.path.join(current_folder, filename)

                with open(save_path, "wb") as f:
                    received_size = 0

                    while received_size < file_size:
                        chunk = conn.recv(min(4096, file_size - received_size))  # **严格按 file_size 读取**
                        if not chunk:
                            break
                        f.write(chunk)
                        received_size += len(chunk)

                    print(f"✅ 文件接收完成: {save_path}, 实际大小: {received_size} 字节")

                # **数据完整性检查**
                if received_size != file_size:
                    print(f"⚠️ 文件大小不匹配！可能数据丢失 ({received_size} != {file_size})")

            except Exception as e:
                print(f"❌ 传输错误: {e}")

            finally:
                conn.close()  # 关闭连接

receive_file()
