import pandas as pd

def load_data(filepath):
    """
    Load dataset from a given csv file.

    Parameters:
    filepath (str): Path to the csv file

    Returns:
    DataFrame: Loaded data
    """
    data = pd.read_csv(filepath)
    return data


from sklearn.preprocessing import LabelEncoder


def preprocess_data(data):
    """
    Preprocess the data.

    Parameters:
    data (DataFrame): The data to be preprocessed

    Returns:
    DataFrame: Preprocessed data
    """
    # Fill missing values with mean
    data.fillna(data.mean(), inplace=True)

    # Convert categorical variables to numeric
    label_encoders = {}
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    return data, label_encoders
