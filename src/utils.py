import os
import joblib
import pandas as pd


# --------------------------------------------------
# 1. Create directory if it doesn't exist
# --------------------------------------------------

def create_directory(directory):
    """
    Create a directory if it does not already exist.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

        print(f"Directory created: {directory}")

    else:
        print(f"Directory already exists: {directory}")


# --------------------------------------------------
# 2. Save a machine learning model
# --------------------------------------------------

def save_model(model, file_path):
    """
    Save a trained machine learning model using joblib.
    """

    directory = os.path.dirname(file_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    joblib.dump(model, file_path)

    print(f"Model saved successfully: {file_path}")


# --------------------------------------------------
# 3. Load a machine learning model
# --------------------------------------------------

def load_model(file_path):
    """
    Load a saved machine learning model.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Model file not found: {file_path}"
        )

    model = joblib.load(file_path)

    print(f"Model loaded successfully: {file_path}")

    return model


# --------------------------------------------------
# 4. Load CSV dataset
# --------------------------------------------------

def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    print(f"Dataset loaded: {file_path}")
    print(f"Dataset shape: {df.shape}")

    return df


# --------------------------------------------------
# 5. Get dataset information
# --------------------------------------------------

def get_dataset_info(df):
    """
    Display basic information about the dataset.
    """

    print("\nDataset Information")
    print("-" * 40)

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn Names:")
    print(list(df.columns))

    print("\nMissing Values:")
    print(df.isnull().sum().sum())

    if "Class" in df.columns:
        print("\nClass Distribution:")
        print(df["Class"].value_counts())


# --------------------------------------------------
# 6. Calculate fraud percentage
# --------------------------------------------------

def calculate_fraud_percentage(df):
    """
    Calculate the percentage of fraudulent transactions.
    """

    if "Class" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'Class' column."
        )

    total_transactions = len(df)

    fraud_transactions = df["Class"].sum()

    fraud_percentage = (
        fraud_transactions / total_transactions
    ) * 100

    return fraud_percentage


# --------------------------------------------------
# 7. Format prediction result
# --------------------------------------------------

def format_prediction(prediction, probability):
    """
    Convert model prediction into a user-friendly result.
    """

    if prediction == 1:
        result = "Fraud"
    else:
        result = "Normal"

    return {
        "prediction": result,
        "fraud_probability": round(
            probability * 100,
            2
        )
    }


# --------------------------------------------------
# 8. Determine risk level
# --------------------------------------------------

def get_risk_level(probability):
    """
    Determine transaction risk based on fraud probability.
    """

    probability = probability * 100

    if probability >= 80:
        return "High Risk"

    elif probability >= 50:
        return "Medium Risk"

    else:
        return "Low Risk"


# --------------------------------------------------
# 9. Create prediction summary
# --------------------------------------------------

def create_prediction_summary(prediction, probability):
    """
    Create a complete prediction summary.
    """

    result = format_prediction(
        prediction,
        probability
    )

    risk_level = get_risk_level(
        probability
    )

    result["risk_level"] = risk_level

    return result

