# 💳 AI-Powered Credit Card Fraud Detection

An end-to-end Machine Learning project that detects fraudulent credit card transactions using a combination of **K-Means Clustering** and **Random Forest Classification**.

The project uses a real-world credit card transaction dataset, performs data preprocessing and exploratory analysis, applies unsupervised learning to identify transaction patterns, and then uses supervised learning to classify transactions as **Fraudulent** or **Normal**. A **Streamlit dashboard** provides an interactive interface for fraud detection and model analysis.

---

## 📌 Project Overview

Credit card fraud is a major challenge in the financial industry. Since fraudulent transactions are rare compared to legitimate transactions, detecting them accurately requires effective data preprocessing and machine learning techniques.

This project combines:

* **K-Means Clustering** – identifies transaction patterns and behavioral groups.
* **Random Forest Classifier** – predicts whether a transaction is fraudulent or normal.
* **Streamlit** – provides an interactive web-based dashboard.
* **Pandas & NumPy** – data processing and analysis.
* **Scikit-learn** – machine learning and evaluation.

### 🔄 Project Workflow

```text
Credit Card Transaction Dataset
            ↓
     Data Preprocessing
            ↓
      Feature Scaling
            ↓
     K-Means Clustering
            ↓
 Cluster + Cluster Distance
            ↓
    Random Forest Model
            ↓
 Fraud / Normal Prediction
            ↓
 Fraud Probability
            ↓
      Risk Classification
            ↓
     Streamlit Dashboard
```

---

## 🎯 Objectives

* Detect fraudulent credit card transactions.
* Handle the highly imbalanced transaction dataset.
* Apply data preprocessing and feature scaling.
* Use K-Means to identify transaction behavior patterns.
* Use Random Forest for fraud classification.
* Generate fraud probability scores.
* Categorize transactions into Low, Medium, and High Risk.
* Provide an interactive Streamlit dashboard.
* Evaluate the model using classification metrics and visualizations.

---

## 🧠 Machine Learning Approach

### 1. Data Preprocessing

The dataset is first cleaned and prepared for machine learning.

Steps include:

* Loading the transaction dataset.
* Checking missing values.
* Removing duplicate transactions.
* Separating features and target variable.
* Splitting data into training and testing sets.
* Scaling `Time` and `Amount` using `StandardScaler`.

### Dataset Statistics

| Description                     |   Value |
| ------------------------------- | ------: |
| Original Records                | 284,807 |
| Original Features               |      30 |
| Target Column                   |   Class |
| Duplicate Rows                  |   1,081 |
| Records After Duplicate Removal | 283,726 |
| Missing Values                  |       0 |

The target variable is:

```text
0 → Normal Transaction
1 → Fraudulent Transaction
```

---

## 🔵 K-Means Clustering

K-Means is an **unsupervised machine learning algorithm** used to group transactions based on their characteristics.

In this project, K-Means is used with:

```text
Number of Clusters = 3
```

For every transaction, the model generates:

* `Cluster`
* `Cluster_Distance`

These features provide additional information about the transaction's behavioral pattern.

### Why K-Means?

Fraudulent transactions can sometimes have different behavioral patterns from normal transactions. K-Means helps identify these patterns before classification.

---

## 🌲 Random Forest Classification

Random Forest is used as the main supervised classification algorithm.

The Random Forest model learns from:

```text
Original Features
       +
K-Means Cluster
       +
Cluster Distance
       ↓
Random Forest
       ↓
Fraud / Normal
```

The model uses:

```text
n_estimators = 100
class_weight = balanced
random_state = 42
```

Using `class_weight="balanced"` helps the model handle the severe class imbalance between normal and fraudulent transactions.

---

## ⚠️ Risk Classification

The model generates a fraud probability for every transaction.

The project categorizes the risk level as:

| Fraud Probability | Risk Level     |
| ----------------: | -------------- |
|             ≥ 80% | 🔴 High Risk   |
|             ≥ 50% | 🟠 Medium Risk |
|             < 50% | 🟢 Low Risk    |

Example:

```text
Fraud Probability = 91.5%
Risk Level = High Risk
```

---

## 📊 Model Evaluation

The project evaluates the Random Forest model using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Precision-Recall AUC
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

Evaluation results and visualizations are stored in:

```text
evaluation/
├── confusion_matrix.png
├── roc_curve.png
├── precision_recall_curve.png
├── classification_report.csv
└── evaluation_summary.csv
```

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit application.

### Dashboard

Displays:

* Total Transactions
* Normal Transactions
* Fraudulent Transactions
* Fraud Detection Rate
* Transaction Overview
* Average Transaction Amount
* K-Means + Random Forest information

### Fraud Detection

Users can upload a CSV containing transaction data.

The application:

1. Validates the uploaded data.
2. Preprocesses the transaction features.
3. Applies K-Means clustering.
4. Calculates cluster distance.
5. Performs Random Forest prediction.
6. Calculates fraud probability.
7. Assigns a risk level.
8. Displays prediction results directly on the screen.

### Analytics

Provides:

* Transaction amount distribution
* Fraud vs Normal transaction distribution
* Fraud statistics
* Average transaction comparison
* Dataset preview

### Model Performance

Displays:

* Random Forest configuration
* K-Means configuration
* ROC-AUC
* Precision-Recall AUC
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

### About Project

Provides information about:

* Project objective
* Technologies used
* Machine learning workflow
* K-Means
* Random Forest
* System architecture

---

## 📂 Project Structure

```text
Credit Card Fraud Detection/
│
├── data/
│   ├── creditcard.csv
│   ├── test_data.csv
│   └── sample_transactions.csv
│
├── src/
│   ├── data.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   └── model_metadata.pkl
│
├── evaluation/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── classification_report.csv
│   └── evaluation_summary.csv
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* Streamlit

### Machine Learning

* K-Means Clustering
* Random Forest Classifier
* StandardScaler

### Development Tools

* VS Code
* Git
* GitHub
* Streamlit

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
```

Navigate into the project:

```bash
cd credit-card-fraud-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1: Train the Models

From the project root directory:

```bash
python src/train.py
```

This generates the trained models inside:

```text
models/
```

### Step 2: Evaluate the Model

```bash
python src/evaluate.py
```

This generates evaluation metrics and graphs inside:

```text
evaluation/
```

### Step 3: Run the Streamlit Application

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

## 📤 Testing Fraud Detection

A small sample dataset is included for testing:

```text
data/sample_transactions.csv
```

It contains a small number of transactions so that the Streamlit prediction interface can be tested quickly.

Go to:

```text
Fraud Detection → Upload CSV
```

and select:

```text
sample_transactions.csv
```

The prediction results are displayed directly on the Streamlit screen.

---

## 📋 Input Features

The prediction system expects the following columns:

```text
Time
V1
V2
V3
V4
V5
V6
V7
V8
V9
V10
V11
V12
V13
V14
V15
V16
V17
V18
V19
V20
V21
V22
V23
V24
V25
V26
V27
V28
Amount
```

The `Class` column is used as the target during training.

For prediction uploads, the model uses the transaction features and generates the prediction itself.

---

## 🔐 Handling Class Imbalance

The dataset contains significantly more normal transactions than fraudulent transactions.

Therefore, simply using accuracy can be misleading.

This project uses:

```python
class_weight="balanced"
```

in the Random Forest classifier and evaluates the model using metrics such as:

* Precision
* Recall
* F1-Score
* ROC-AUC
* Precision-Recall AUC

These metrics provide a better understanding of fraud detection performance.

---

## 🚀 Future Improvements

Possible improvements include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Trying XGBoost or other advanced ensemble models.
* Implementing SMOTE for class imbalance handling.
* Adding real-time transaction monitoring.
* Adding database integration.
* Adding user authentication.
* Deploying the Streamlit application to the cloud.
* Adding explainable AI using SHAP.
* Adding automated fraud alerts.
* Improving the dashboard with additional visualizations.

---

## 💡 Key Learning Outcomes

Through this project, I gained practical experience in:

* Data cleaning and preprocessing
* Handling imbalanced datasets
* Exploratory Data Analysis
* Unsupervised Machine Learning
* Supervised Machine Learning
* K-Means clustering
* Random Forest classification
* Feature engineering
* Model evaluation
* Model serialization using Joblib
* Streamlit application development
* Git and GitHub project management

---

## 👩‍💻 Author

**N.Sai Prasanthi**

B.Tech – Computer Science and Engineering

### Skills Demonstrated

```text
Python | Machine Learning | Pandas | NumPy |
Scikit-learn | Streamlit | Git | GitHub
```

---

## ⭐ Project Highlights

> **An end-to-end AI-powered fraud detection system that combines K-Means clustering with Random Forest classification to identify suspicious credit card transactions and provide probability-based risk levels through an interactive Streamlit dashboard.**

If you find this project useful, consider giving the repository a ⭐.
