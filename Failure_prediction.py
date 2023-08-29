import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import requests
import re



def predict_failure_solution(failure):
    # Load the dataset
    data = pd.read_csv('Jenkins_log_failure_dataset.csv')
    df = pd.DataFrame(data)
    # Convert text to feature vectors
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['FAILURE'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, df['POSSIBLE_SOLUTION'], test_size=0.2)

    # Train a Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Predict on a new input string
    new_input_vector = vectorizer.transform([failure])
    predicted_label = clf.predict(new_input_vector)

    print(f"Predicted label for '{failure}': {predicted_label}")


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
    with open(log_file, 'r') as file:
        log_data = file.read()

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
        print(test_case_range)

        # Define regular expressions for parsing
        pattern = r'.*Error.*'  # Example: [ERROR] Some error message
        error_messages = re.findall(pattern, test_case_range)

        # Print the failure messages
        print("\nFailure Messages:")
        for message in error_messages:
            print(message)
            predict_failure_solution(message)
        print("\n")



jenkins_url = "http://localhost:8080/job/Test_Job/lastBuild/consoleText"
log_file_path = "jenkins.log"

download_console_log(jenkins_url, log_file_path)
parse_console_log(log_file_path)
