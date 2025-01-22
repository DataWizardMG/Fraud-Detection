from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import mlflow

def print_performance_metrics(y_true, y_pred, model):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"{model} Accuracy: {accuracy * 100:.2f}%")
    print(f"{model} Precision: {precision * 100:.2f}%")
    print(f"{model} Recall: {recall * 100:.2f}%")
    print(f"{model} F1-Score: {f1 * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)

    mlflow.log_metric("accuracy", accuracy*100)
    mlflow.log_metric("precision", precision*100)
    mlflow.log_metric("recall", recall*100)
    mlflow.log_metric("f1_score", f1*100)

def logistic_regression_pipeline(x_train, y_train, x_test, y_test):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    print_performance_metrics(y_test, y_pred, 'Logistic Regression')

def ada_boost_pipeline(x_train, y_train, x_test, y_test):
    ada = AdaBoostClassifier(n_estimators=200, random_state=42)
    ada.fit(x_train, y_train)
    y_pred = ada.predict(x_test)
    print_performance_metrics(y_test, y_pred, 'AdaBoost')
    return ada  # Return the trained model

def save_best_model(model, filepath):
    """Save the best-performing model to a file."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")