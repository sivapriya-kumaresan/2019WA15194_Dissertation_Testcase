import os
try:
    import requests
except:
    os.system("python -m pip install requests")
    import requests


HOST = 'ios-xe-mgmt-latest.cisco.com'
PORT = 443
USER = 'admin'
PASS = 'C1sco12345'

# disable urlib3 warning
requests.packages.urllib3.disable_warnings()

headers = {'Content-Type': 'application/yang-data+json',
'Accept': 'application/yang-data+json'}

url = f"https://{HOST}/restconf/data/Cisco-IOS-XE-interfaces-oper:interfaces/interface=GigabitEthernet1/admin-status"
response = requests.get(url, auth=(USER, PASS), headers=headers, verify=False)
print(response.text)
if "if-state-down" in response.text:
  print("Verification Success")
  return True
else:
  print("Expected admin-status if-state-down not found in response")
  return False
