import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

DATA_PATH = "data/creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Remove Duplicate Rows
# --------------------------------------------------

print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)


# --------------------------------------------------
# 3. Separate Features and Target
# --------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# --------------------------------------------------
# 4. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())


# --------------------------------------------------
# 5. Scale Time and Amount
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

columns_to_scale = ["Time", "Amount"]

X_train_scaled[columns_to_scale] = scaler.fit_transform(
    X_train[columns_to_scale]
)

X_test_scaled[columns_to_scale] = scaler.transform(
    X_test[columns_to_scale]
)


# --------------------------------------------------
# 6. K-Means Clustering
# --------------------------------------------------

print("\nTraining K-Means clustering model...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_train_scaled)

# Cluster labels
train_clusters = kmeans.predict(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

# Distance from nearest cluster center
train_distances = kmeans.transform(X_train_scaled).min(axis=1)
test_distances = kmeans.transform(X_test_scaled).min(axis=1)


# --------------------------------------------------
# 7. Add K-Means Features
# --------------------------------------------------

X_train_ml = X_train_scaled.copy()
X_test_ml = X_test_scaled.copy()

X_train_ml["Cluster"] = train_clusters
X_test_ml["Cluster"] = test_clusters

X_train_ml["Cluster_Distance"] = train_distances
X_test_ml["Cluster_Distance"] = test_distances

print("\nK-Means features added:")
print("Cluster")
print("Cluster_Distance")


# --------------------------------------------------
# 8. Train Random Forest
# --------------------------------------------------

print("\nTraining Random Forest classifier...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train_ml, y_train)

print("Random Forest training completed.")


# --------------------------------------------------
# 9. Evaluate Model
# --------------------------------------------------

y_probability = rf_model.predict_proba(X_test_ml)[:, 1]

roc_auc = roc_auc_score(y_test, y_probability)

pr_auc = average_precision_score(y_test, y_probability)

print("\nModel Performance")
print("----------------------")
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))


# --------------------------------------------------
# 10. Create Models Folder
# --------------------------------------------------

os.makedirs("models", exist_ok=True)


# --------------------------------------------------
# 11. Save Random Forest
# --------------------------------------------------

joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)

print("\nRandom Forest saved:")
print("models/random_forest.pkl")


# --------------------------------------------------
# 12. Save Scaler
# --------------------------------------------------

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaler saved:")
print("models/scaler.pkl")


# --------------------------------------------------
# 13. Save K-Means
# --------------------------------------------------

joblib.dump(
    kmeans,
    "models/kmeans.pkl"
)

print("K-Means saved:")
print("models/kmeans.pkl")


# --------------------------------------------------
# 14. Save Test Data
# --------------------------------------------------

test_data = X_test.copy()
test_data["Class"] = y_test.values

os.makedirs("data", exist_ok=True)

test_data.to_csv(
    "data/test_data.csv",
    index=False
)

print("\nTest data saved:")
print("data/test_data.csv")


# --------------------------------------------------
# 15. Save Metadata
# --------------------------------------------------

metadata = {
    "model_name": "AI-Powered Credit Card Fraud Detection",
    "algorithm": "K-Means Clustering + Random Forest",
    "kmeans_clusters": 3,
    "random_forest_estimators": 100,
    "random_state": 42,
    "class_weight": "balanced",
    "features": list(X.columns),
    "additional_features": [
        "Cluster",
        "Cluster_Distance"
    ],
    "feature_count": len(X.columns) + 2,
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

joblib.dump(
    metadata,
    "models/model_metadata.pkl"
)

print("\nMetadata saved:")
print("models/model_metadata.pkl")

print("\n===================================")
print("MODEL TRAINING COMPLETED")
print("===================================")