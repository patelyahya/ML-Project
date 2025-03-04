import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self):
        self.model = None

    def train_model(self, X_train, y_train):
        with mlflow.start_run():
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            mlflow.log_param("n_estimators", 100)
            mlflow.sklearn.log_model(self.model, "model")
            
            return self.model

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        return accuracy, precision, recall

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    trainer = ModelTrainer()
    model = trainer.train_model(X_train, y_train)
    accuracy, precision, recall = trainer.evaluate_model(X_test, y_test)
    print(f"Model trained with accuracy: {accuracy}, precision: {precision}, recall: {recall}")
