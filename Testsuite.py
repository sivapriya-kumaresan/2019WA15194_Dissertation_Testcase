# Testsuite to execute all testcases
import subprocess
import time

def main():
    for i in range(1,2):
        print(f"Running Testcase{i}.py",flush=True)
        testcase = f"Testcase{i}.py"
        subprocess.run(["python", testcase,str(i)])
        print(f"Testcase{i} execution complete")
        print("Wait for 10 seconds before executing next testcase")
        time.sleep(10)
# Execute Testcases
if __name__ == "__main__":
    main()
