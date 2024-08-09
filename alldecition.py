import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

def runscript(csv_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Replace specific values with 'No'
    df = df.replace({'No phone service': 'No', 'No internet service': 'No'})

    # Convert string values to integers or floats, or factorize if not possible
    def str_to_int(data):
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = data[col].astype(float)
                except ValueError:
                    data[col], _ = pd.factorize(data[col])
        return data

    df = str_to_int(df)

    # Remove rows with empty strings
    df.replace(" ", np.nan, inplace=True)
    df.dropna(inplace=True)

    # Sample data if it's larger than 1300 rows
    if len(df) > 1300:
        df = df.sample(n=1300, random_state=42)

    # Split the data into features and target variable
    X = df.iloc[:, :-1]  # All columns except the last
    y = df.iloc[:, -1]   # Only the last column

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the decision tree model
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Plot the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(clf, filled=True, node_ids=True, impurity=False, feature_names=X.columns, max_depth=5, fontsize=10)
    plt.title("Decision Tree")
    decision_tree_plot_path = f"{os.path.splitext(csv_file_path)[0]}_decision_tree_plot.png"
    plt.savefig(decision_tree_plot_path)
    plt.close()

    # Plot feature importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(20, 10))
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    importance_plot_path = f"{os.path.splitext(csv_file_path)[0]}_importance.png"
    plt.savefig(importance_plot_path)
    plt.close()

    return accuracy, decision_tree_plot_path, importance_plot_path

def process_all_csv_files(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return []

    results = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            csv_file_path = os.path.join(directory_path, filename)
            try:
                accuracy, decision_tree_plot_path, importance_plot_path = runscript(csv_file_path)
                results.append((filename, accuracy, decision_tree_plot_path, importance_plot_path))
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    return results

# Example usage
directory_path = r'C:\Users\YourUsername\Documents\csv_files'  # Update this to your actual directory path
results = process_all_csv_files(directory_path)

for result in results:
    print(f"File: {result[0]}, Accuracy: {result[1]}")
    print(f"Decision Tree Plot: {result[2]}")
    print(f"Feature Importances Plot: {result[3]}")
