import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def clean_data(self, data):
        # Remove missing values
        data = data.dropna()
        return data

    def feature_engineering(self, data):
        # Convert categorical variables to dummy variables
        data = pd.get_dummies(data)
        return data

    def split_data(self, data, target_column):
        # Split data into features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    preprocessor = DataPreprocessor()
    data = preprocessor.clean_data(data)
    data = preprocessor.feature_engineering(data)
    X_train, X_test, y_train, y_test = preprocessor.split_data(data, 'target')
    print("Data preprocessing completed.")
