import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------

file_path = "Data/creditcard.csv"

df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)


# ---------------------------------------
# 2. Check Missing Values
# ---------------------------------------

print("\nMissing values:")
print(df.isnull().sum().sum())


# ---------------------------------------
# 3. Remove Duplicate Rows
# ---------------------------------------

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)


# ---------------------------------------
# 4. Separate Features and Target
# ---------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ---------------------------------------
# 5. Train/Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ---------------------------------------
# 6. Check Class Distribution
# ---------------------------------------

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ---------------------------------------
# 7. Feature Scaling
# ---------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nScaled training shape:", X_train_scaled.shape)
print("Scaled testing shape:", X_test_scaled.shape)