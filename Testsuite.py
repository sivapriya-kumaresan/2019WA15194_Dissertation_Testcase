# Testsuite to execute all testcases
import subprocess
import time

def run_script(testcase_name):
    subprocess.run(["python", testcase_name])

def main():
    for i in range(1,3):
        print(f"Running Testcase{i}.py")
        testcase = f"Testcase{i}.py"
        run_script(testcase)
        print(f"Testcase{i} execution complete")
        print("Wait for 10 seconds before executing next testcase")
        time.sleep(10)
# Execute Testcases
if __name__ == "__main__":
    main()
