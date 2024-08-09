import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def runscript(csv_file_path):
    # Read the CSV file
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
    def remove_rows_with_empty_strings(df):
        mask = df.applymap(lambda x: x == ' ')
        rows_with_empty_strings = mask.any(axis=1)
        df_cleaned = df[~rows_with_empty_strings]
        return df_cleaned

    df = remove_rows_with_empty_strings(df)

    # Assuming we want a sample of 1300 rows, otherwise use the entire dataframe
    if len(df) > 1300:
        dfn = df.sample(n=1300, random_state=42)
    else:
        dfn = df

    # Split the data into features and target variable
    X = dfn.drop(dfn.columns[-1], axis=1)  # Assuming the target is the last column
    y = dfn[df.columns[-1]]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest classifier
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_clf.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Plot the decision tree of the first estimator in the forest
    estimator = rf_clf.estimators_[0]
    plt.figure(figsize=(20, 10))
    plot_tree(estimator, filled=True, feature_names=X.columns, class_names=np.unique(y).astype(str), rounded=True, proportion=True)
    tree_plot_path = f"{os.path.splitext(csv_file_path)[0]}_rf_decision_tree_plot.png"
    plt.savefig(tree_plot_path)
    plt.close()

    # Plot feature importances
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    importance_plot_path = f"{os.path.splitext(csv_file_path)[0]}_rf_feature_importances.png"
    plt.savefig(importance_plot_path)
    plt.close()

    return accuracy, tree_plot_path, importance_plot_path

def process_all_csv_files(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return []

    results = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            csv_file_path = os.path.join(directory_path, filename)
            try:
                accuracy, tree_plot_path, importance_plot_path = runscript(csv_file_path)
                results.append((filename, accuracy, tree_plot_path, importance_plot_path))
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
