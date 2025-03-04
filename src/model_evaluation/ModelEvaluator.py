import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelEvaluator:
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
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
    model = mlflow.sklearn.load_model("random_forest_model")
    evaluator = ModelEvaluator()
    accuracy, precision, recall = evaluator.evaluate_model(model, X, y)
    print(f"Model evaluated with accuracy: {accuracy}, precision: {precision}, recall: {recall}")
