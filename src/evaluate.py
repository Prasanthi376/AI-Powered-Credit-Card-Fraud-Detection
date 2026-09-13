import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)


# --------------------------------------------------
# Load Test Data
# --------------------------------------------------

test_data = pd.read_csv(
    "data/test_data.csv"
)

X_test = test_data.drop(
    "Class",
    axis=1
)

y_test = test_data["Class"]


# --------------------------------------------------
# Load Models
# --------------------------------------------------

model = joblib.load(
    "models/random_forest.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

kmeans = joblib.load(
    "models/kmeans.pkl"
)


# --------------------------------------------------
# Scale Data
# --------------------------------------------------

X_test_scaled = X_test.copy()

columns_to_scale = [
    "Time",
    "Amount"
]

X_test_scaled[columns_to_scale] = scaler.transform(
    X_test[columns_to_scale]
)


# --------------------------------------------------
# K-Means
# --------------------------------------------------

clusters = kmeans.predict(
    X_test_scaled
)

cluster_distances = kmeans.transform(
    X_test_scaled
).min(axis=1)


# --------------------------------------------------
# Add K-Means Features
# --------------------------------------------------

X_test_scaled["Cluster"] = clusters

X_test_scaled["Cluster_Distance"] = cluster_distances


# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred = model.predict(
    X_test_scaled
)

y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()


# --------------------------------------------------
# Print Results
# --------------------------------------------------

print("\nClassification Report")
print("----------------------")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# ROC-AUC
# --------------------------------------------------

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nROC-AUC:", round(roc_auc, 4))


# --------------------------------------------------
# PR-AUC
# --------------------------------------------------

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("PR-AUC:", round(pr_auc, 4))


# --------------------------------------------------
# Create Evaluation Folder
# --------------------------------------------------

os.makedirs(
    "evaluation",
    exist_ok=True
)


# --------------------------------------------------
# Save Classification Report
# --------------------------------------------------

report_df.to_csv(
    "evaluation/classification_report.csv"
)


# --------------------------------------------------
# Confusion Matrix Plot
# --------------------------------------------------

plt.figure()

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.colorbar()

plt.savefig(
    "evaluation/confusion_matrix.png"
)

plt.close()


# --------------------------------------------------
# ROC Curve
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure()

plt.plot(
    fpr,
    tpr
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.savefig(
    "evaluation/roc_curve.png"
)

plt.close()


# --------------------------------------------------
# Precision-Recall Curve
# --------------------------------------------------

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_probability
)

plt.figure()

plt.plot(
    recall,
    precision
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve"
)

plt.savefig(
    "evaluation/precision_recall_curve.png"
)

plt.close()


# --------------------------------------------------
# Save Evaluation Summary
# --------------------------------------------------

summary = pd.DataFrame({
    "Metric": [
        "ROC-AUC",
        "PR-AUC"
    ],
    "Score": [
        roc_auc,
        pr_auc
    ]
})

summary.to_csv(
    "evaluation/evaluation_summary.csv",
    index=False
)


print("\nEvaluation files saved successfully.")

print("\n===================================")
print("MODEL EVALUATION COMPLETED")
print("===================================")