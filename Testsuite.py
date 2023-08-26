# Testsuite to execute all testcases
import os
import time
import tabulate
testcase_result = [["TESTCASE TITLE " ,"RESULT"]]

def main():
    for i in range(1,2):
        print("="*100)
        print(f"Running Testcase{i}.py",flush=True)
        time.sleep(5)
        testcase = f"Testcase{i}.py"
        status = os.system(f"python {testcase}")
        print(f"Testcase{i} execution complete")
        print("Wait for 10 seconds before executing next testcase")
        time.sleep(10)
        print("="*100)
        result = "PASS" if not status else "FAIL"
        print("Testcase name : ",testcase)
        print("Testcase result : ",result)
        testcase_result.append([testcase,result])
        print("="*100)
   result_table = tabulate(testcase_result,headers='firstrow',tablefmt='grid') 
   print(result_table)
# Execute Testcases
if __name__ == "__main__":
    main()
