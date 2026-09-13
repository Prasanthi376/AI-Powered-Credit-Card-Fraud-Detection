import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------

file_path = "Data/creditcard.csv"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)


# ---------------------------------------
# 2. Class Distribution
# ---------------------------------------

print("\nClass distribution:")
print(df["Class"].value_counts())

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Class")

plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

plt.show()


# ---------------------------------------
# 3. Transaction Amount Distribution
# ---------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Amount",
    bins=50
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.show()


# ---------------------------------------
# 4. Amount by Class
# ---------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.title("Transaction Amount by Class")
plt.xlabel("Class")
plt.ylabel("Amount")

plt.show()


# ---------------------------------------
# 5. Amount Statistics
# ---------------------------------------

print("\nAverage transaction amount:")
print(df.groupby("Class")["Amount"].mean())

print("\nMedian transaction amount:")
print(df.groupby("Class")["Amount"].median())


# ---------------------------------------
# 6. Transaction Time
# ---------------------------------------

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Time",
    bins=50
)

plt.title("Transaction Time Distribution")
plt.xlabel("Time")
plt.ylabel("Number of Transactions")

plt.show()


# ---------------------------------------
# 7. Fraud Transactions Over Time
# ---------------------------------------

fraud_df = df[df["Class"] == 1]

plt.figure(figsize=(10, 5))

sns.histplot(
    data=fraud_df,
    x="Time",
    bins=50
)

plt.title("Fraudulent Transactions Over Time")
plt.xlabel("Time")
plt.ylabel("Number of Fraud Transactions")

plt.show()


# ---------------------------------------
# 8. Correlation with Class
# ---------------------------------------

correlation = df.corr(numeric_only=True)["Class"].sort_values()

print("\nCorrelation with Class:")
print(correlation)