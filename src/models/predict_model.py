from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model.

    Parameters:
    model (RandomForestClassifier): The model to be evaluated
    X_test (DataFrame): The test features
    y_test (Series): The test target

    Returns:
    None
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)
    return report