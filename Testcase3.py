import os
try:
    import paramiko 
except:
    os.system("python -m pip install paramiko")
    import paramiko 
  
ssh_username = "127.0.0.1"
ssh_password = "admin"
ssh_server = "localhost"

ssh_client = paramiko.SSHClient()

ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh_client.connect(hostname=ssh_server, username=ssh_username, password=ssh_password)

command = "ls -l"
stdin, stdout, stderr = ssh_client.exec_command(command)

print("Command Output:")
print(stdout.read().decode("utf-8"))

ssh_client.close()
