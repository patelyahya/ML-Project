import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def preprocess_data(data):
    # Data cleaning and feature engineering
    data = data.dropna()
    data = pd.get_dummies(data)
    X = data.drop('target', axis=1)
    y = data['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.sklearn.log_model(model, "model")
        
        return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    return accuracy, precision, recall

def save_model(model, model_name):
    mlflow.sklearn.save_model(model, model_name)

def load_model(model_name):
    return mlflow.sklearn.load_model(model_name)

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    accuracy, precision, recall = evaluate_model(model, X_test, y_test)
    save_model(model, "random_forest_model")
    print(f"Model saved with accuracy: {accuracy}, precision: {precision}, recall: {recall}")
