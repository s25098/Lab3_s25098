import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()

    print(f"Data after dropping missing values: {data.shape}")
    return data


def preprocess_data(data):
    le = LabelEncoder()
    categorical_columns = ['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'region']
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])

    X = data.drop(['score', 'rownames'], axis=1)
    y = data['score']

    numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()

    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    data = load_and_clean_data("CollegeDistance.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
