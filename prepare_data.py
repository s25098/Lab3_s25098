import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns


def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)

    data = data.dropna()

    print(f"Data after dropping missing values: {data.shape}")
    return data


def explore_data(data):
    data = data.drop(columns=['rownames'], errors='ignore')

    data = pd.get_dummies(data, drop_first=True)

    correlation_matrix = data.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f")
    plt.title("Correlation Matrix", fontsize=18, fontweight='bold')
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.histplot(data['score'], bins=30, kde=True, color='darkblue')
    plt.title("Dystrybucja zmiennej 'score'", fontsize=16, fontweight='bold')
    plt.xlabel("Score", fontsize=14)
    plt.ylabel("Częstotliwość", fontsize=14)
    plt.show()

    return data


def preprocess_data(data):
    X = data.drop(['score'], axis=1)
    y = data['score']

    categorical_columns = X.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        X[column] = LabelEncoder().fit_transform(X[column])

    X['distance_tuition'] = X['distance'] * X['tuition']
    X['log_income'] = np.log(X['income'].replace(0, 1))
    X['log_distance'] = np.log(X['distance'].replace(0, 1))

    numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test):
    if not os.path.exists('data'):
        os.makedirs('data')

    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False, header=True)
    y_test.to_csv("data/y_test.csv", index=False, header=True)


if __name__ == "__main__":
    data = load_and_clean_data("CollegeDistance.csv")

    explore_data(data)

    X_train, X_test, y_train, y_test = preprocess_data(data)

    y_train = pd.DataFrame(y_train, columns=["score"])
    y_test = pd.DataFrame(y_test, columns=["score"])

    save_preprocessed_data(X_train, X_test, y_train, y_test)
