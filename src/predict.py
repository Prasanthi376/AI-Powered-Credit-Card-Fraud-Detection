import joblib
import pandas as pd


# --------------------------------------------------
# Load Models
# --------------------------------------------------

model = joblib.load("models/random_forest.pkl")

scaler = joblib.load("models/scaler.pkl")

kmeans = joblib.load("models/kmeans.pkl")

metadata = joblib.load("models/model_metadata.pkl")


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_transaction(transaction):

    # Convert input to DataFrame
    df = pd.DataFrame([transaction])

    # Expected original features
    expected_features = [
        "Time"
    ]

    expected_features += [f"V{i}" for i in range(1, 29)]

    expected_features += ["Amount"]

    # Check features
    missing_features = [
        feature
        for feature in expected_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Keep correct order
    df = df[expected_features]

    # --------------------------------------------------
    # Scaling
    # --------------------------------------------------

    columns_to_scale = ["Time", "Amount"]

    df_scaled = df.copy()

    df_scaled[columns_to_scale] = scaler.transform(
        df[columns_to_scale]
    )

    # --------------------------------------------------
    # K-Means
    # --------------------------------------------------

    cluster = kmeans.predict(df_scaled)[0]

    cluster_distance = kmeans.transform(
        df_scaled
    ).min(axis=1)[0]

    # --------------------------------------------------
    # Add K-Means Features
    # --------------------------------------------------

    df_scaled["Cluster"] = cluster

    df_scaled["Cluster_Distance"] = cluster_distance

    # --------------------------------------------------
    # Random Forest Prediction
    # --------------------------------------------------

    prediction = model.predict(df_scaled)[0]

    probability = model.predict_proba(
        df_scaled
    )[0][1]

    # --------------------------------------------------
    # Risk Level
    # --------------------------------------------------

    if probability >= 0.80:
        risk_level = "High Risk"

    elif probability >= 0.50:
        risk_level = "Medium Risk"

    else:
        risk_level = "Low Risk"

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability),
        "risk_level": risk_level,
        "cluster": int(cluster),
        "cluster_distance": float(cluster_distance)
    }


# --------------------------------------------------
# Test Prediction
# --------------------------------------------------

if __name__ == "__main__":

    sample_transaction = {
        "Time": 10000,
        "Amount": 149.62
    }

    # Add V1-V28
    for i in range(1, 29):
        sample_transaction[f"V{i}"] = 0

    result = predict_transaction(
        sample_transaction
    )

    print("\nPrediction Result")
    print("----------------------")

    print(
        "Prediction:",
        "Fraud" if result["prediction"] == 1 else "Normal"
    )

    print(
        "Fraud Probability:",
        round(result["fraud_probability"] * 100, 2),
        "%"
    )

    print(
        "Risk Level:",
        result["risk_level"]
    )

    print(
        "K-Means Cluster:",
        result["cluster"]
    )

    print(
        "Cluster Distance:",
        round(result["cluster_distance"], 4)
    )