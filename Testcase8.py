import os,sys
try:
    import netmiko
    import paramiko
except:
    os.system("python -m pip install netmiko")
    os.system("python -m pip install paramiko")
    import netmiko
    import paramiko

from netmiko import ConnectHandler
from paramiko import SSHException

# Define Cisco device details with incorrect credentials
device = {
    'device_type': 'cisco_ios',
    'ip': 'ios-xe-mgmt-latest.cisco.com',
    'username': 'admin',
    'password': 'Cisco12345',
}

try:
    # Attempt to establish SSH connection to the device
    net_connect = ConnectHandler(**device)

    # If the connection is successful, this won't be reached
    print("Connected to the device successfully.")

except SSHException as e:
    print(f"Authentication Error: {str(e)}")

except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")

finally:
    # Close the SSH connection if it was established
    if 'net_connect' in locals():
        net_connect.disconnect()
    else:
        sys.exit(1)
