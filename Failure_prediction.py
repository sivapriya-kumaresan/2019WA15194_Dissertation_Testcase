import os

package_to_install = ['pandas','scikit-learn','requests','regex','tabulate','matplotlib','openpyxl']
for package_name in package_to_install:
    try:
        __import__(package_name)
    except ImportError:
        os.system(f"python -m pip install {package_name}")
        
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score,recall_score, f1_score, accuracy_score, confusion_matrix,ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
import openpyxl
import requests
import sklearn
import re, matplotlib
from  tabulate import tabulate

def print_result(result_list):
    headers = result_list[0]
    data = result_list[1:]
    table = tabulate(data, headers, tablefmt="simple",stralign="left")
    #table = table.encode("utf-8")
    print(table)

def export_result(result,excel_file):
    df = pd.DataFrame(result[1:], columns=result[0])
    df.to_excel(excel_file, index=False)
    print(f'Result Data has been written to {excel_file}.')

def predict_failure_solution(failure):
    # Load the dataset
    data = pd.read_csv('Jenkins_log_failure_dataset.csv',on_bad_lines='skip')
    df = pd.DataFrame(data)
    # Convert text to feature vectors
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['FAILURE'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, df['POSSIBLE_SOLUTION'], test_size=0.5)

    # Train a Decision Tree Classifier
    clf = DecisionTreeClassifier()
    #clf.fit(X_train, y_train)
    clf.fit(X, df['POSSIBLE_SOLUTION'])
    
    # Predict on a new input string
    new_input_vector = vectorizer.transform(failure)
    predicted_label = clf.predict(new_input_vector)

    print(len(y_test))
    print("X test ")
    for train,test,fail in zip(X_train,X_test,new_input_vector):
        print("Train")
        print(train)
        print("Test")
        print(test)
        print("Fail")
        print(fail)

    print("PRed Data")
    for fail in new_input_vector:
        print(fail)

    
    #Calculate the aacuracy
    accuracy = accuracy_score(y_test, predicted_label)
    print('Accuracy: %.2f' % (accuracy * 100))
    
    return predicted_label

def download_console_log(url, output_file):
    # Jenkins Credentials
    username = 'sivapriya'
    password = 'Priya@1999'

    # Set up the authentication
    auth = (username, password)

    # Send a GET request to the URL with authentication
    response = requests.get(url, auth=auth)

    if response.status_code == 200:
        with open(output_file, 'wb') as file:
            file.write(response.content)
        print("Console log downloaded successfully.")
    else:
        print("Failed to download console log.")


def parse_console_log(log_file):
    with open(log_file, 'r', encoding='utf-8') as file:
        log_data = file.read()

    failures , test_list = [] , []
    results = [["TESTCASE NAME ", "FAILURE", "CAUSE AND SOLUTION"]]
    start_marker = r'Running Testcase.*'
    end_marker = r'.*execution complete'

    start_matches = re.finditer(start_marker, log_data)
    end_matches = re.finditer(end_marker, log_data)

    # Extract start and end positions of test cases
    start_positions = [match.start() for match in start_matches]
    end_positions = [match.end() for match in end_matches]

    # Identify individual testcase
    print("Test Case Ranges:")
    for i in range(len(start_positions)):
        start_pos = start_positions[i]
        end_pos = end_positions[i]
        test_case_range = log_data[start_pos:end_pos]
        testcase_name = re.findall("Running.*", test_case_range)
        print("Testcase name ", testcase_name[0].split(" ")[-1])
        testcase_name = testcase_name[0].split(" ")[-1]
        test_list.append(testcase_name)
        print(test_case_range)

        # Define regular expressions for parsing
        pattern = r'.*Error.*|.*Errno.*|.*not found.*'  # Example: [ERROR] Some error message
        error_messages = re.findall(pattern, test_case_range)

        # Print the failure messages
        print("\nFailure Messages:")
        for message in error_messages:
            print("-"*50)
            print(message)
            failures.append(message)
            print("-"*50)
    prediction = predict_failure_solution(failures)
        
    for test,message,pred in zip(test_list,failures,prediction):
        results.append([test, message, pred])
        
    print_result(results)
    print("\n")
    print("=="*60)
    #print("Accuracy score : ",accuracy)
    export_result(results,os.getcwd()+'\Prediction_results.xlsx')
    
jenkins_url = "http://localhost:8080/job/Test_Job/lastBuild/consoleText"
log_file_path = "jenkins.log"

download_console_log(jenkins_url, log_file_path)
parse_console_log(log_file_path)
