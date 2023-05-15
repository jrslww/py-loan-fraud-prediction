from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model(data, target_column):
    """
    Train a model using the given DataFrame.

    Parameters:
    data (DataFrame): The DataFrame to be used for training
    target_column (str): The target variable column

    Returns:
    RandomForestClassifier: Trained model
    """
    if target_column in data.columns:
        X = data.drop(target_column, axis=1)
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
        model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
        model.fit(X_train, y_train)

        return model, X_test, y_test
    else:
        print(f"Target column {target_column} not found in data.")
        return None, None, None
