# Testsuite to execute all testcases
import subprocess
import time

def run_script(self,testcase_name):
    subprocess.run(["python", testcase_name])

for i in range(1,3)
print(f"Running Testcase{script}.py")
testcase = f"Testcase{i}.py"
run_script(testcase)
print(f"Testcase{script} execution complete")
print("Wait for 10 seconds before executing next testcase")
time.sleep(10)
