import paramiko
from scp import SCPClient

def create_ssh_client(server, port, user, password):
    """创建SSH客户端并连接到服务器."""
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def ensure_remote_path_exists_and_writable(ssh_client, remote_path):
    """确保远程路径存在并且具有写权限."""
    stdin, stdout, stderr = ssh_client.exec_command(f'mkdir -p {remote_path} && touch {remote_path}/testfile && rm {remote_path}/testfile')
    error = stderr.read().decode()
    if error:
        raise PermissionError(f"无法确保远程路径存在或无写权限: {error}")

def transfer_file(local_file, remote_path, server, port, user, password):
    """通过SCP将本地文件传输到远程服务器."""
    ssh_client = create_ssh_client(server, port, user, password)
    
    try:
        ensure_remote_path_exists_and_writable(ssh_client, remote_path)
        with SCPClient(ssh_client.get_transport()) as scp:
            scp.put(local_file, remote_path)
            print(f"文件 {local_file} 已成功传输到 {server}:{remote_path}")
    except Exception as e:
        print(f"文件传输失败: {e}")
    finally:
        ssh_client.close()

# 使用示例
server = "60.204.226.214"
port = 22  # 一般SSH使用22端口
user = "root"
password = "Zhuxinyu13579"
local_file = "/GPFS/data/xinyuzhu-1/FL/data/generate_data.py"  # 例如 "/path/to/local/file.txt"
remote_path = "/data/test/"  # 例如 "/path/to/remote/destination/"

transfer_file(local_file, remote_path, server, port, user, password)
