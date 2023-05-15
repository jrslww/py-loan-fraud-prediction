from sklearn.preprocessing import StandardScaler


def feature_engineering(data):
    """
    Perform feature engineering on the data.

    Parameters:
    data (DataFrame): The data to be processed

    Returns:
    DataFrame: Processed data
    """
    # Assume that 'amount' is a feature in your data
    # We'll standardize it for better performance of our ML model
    if 'amount' in data.columns:
        scaler = StandardScaler()
        data['amount'] = scaler.fit_transform(data['amount'].values.reshape(-1, 1))
        return data, scaler
    else:
        print("Column 'amount' not found in data.")
        return data, None