import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Powered Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

KMEANS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "kmeans.pkl"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_metadata.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "creditcard.csv"
)

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    kmeans = joblib.load(
        KMEANS_PATH
    )

    metadata = joblib.load(
        METADATA_PATH
    )

    return model, scaler, kmeans, metadata


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(
        DATA_PATH
    )

    return df


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    KMEANS_PATH,
    METADATA_PATH
]

missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    st.error(
        "Required model files are missing."
    )

    st.write(
        "Please run the training script first:"
    )

    st.code(
        "python src/train.py"
    )

    st.stop()


# ============================================================
# LOAD MODELS AND DATA
# ============================================================

try:

    model, scaler, kmeans, metadata = load_models()

except Exception as e:

    st.error(
        "Error loading model files."
    )

    st.exception(e)

    st.stop()


try:

    df = load_dataset()

except Exception as e:

    st.error(
        "Error loading dataset."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "💳 Fraud Detection"
)

st.sidebar.markdown(
    """
    ### AI-Powered System

    **Machine Learning Pipeline**

    🔹 K-Means Clustering  
    🔹 Random Forest  
    🔹 Fraud Probability  
    🔹 Risk Classification
    """
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 Fraud Detection",
        "📊 Analytics",
        "🤖 Model Performance",
        "ℹ️ About Project"
    ]
)


# ============================================================
# COMMON DATA
# ============================================================

total_transactions = len(df)

fraud_transactions = int(
    df["Class"].sum()
)

normal_transactions = (
    total_transactions -
    fraud_transactions
)

fraud_rate = (
    fraud_transactions /
    total_transactions
) * 100


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "💳 AI-Powered Credit Card Fraud Detection"
    )

    st.markdown(
        """
        ### Machine Learning based fraud detection using
        **K-Means Clustering + Random Forest**
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

    with col2:

        st.metric(
            "Normal Transactions",
            f"{normal_transactions:,}"
        )

    with col3:

        st.metric(
            "Fraud Transactions",
            f"{fraud_transactions:,}"
        )

    with col4:

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # TRANSACTION OVERVIEW
    # --------------------------------------------------------

    st.subheader(
        "📈 Transaction Overview"
    )

    overview_data = pd.DataFrame(
        {
            "Transaction Type": [
                "Normal",
                "Fraud"
            ],
            "Count": [
                normal_transactions,
                fraud_transactions
            ]
        }
    )

    st.bar_chart(
        overview_data.set_index(
            "Transaction Type"
        )
    )

    # --------------------------------------------------------
    # AMOUNT INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "💰 Transaction Amount Analysis"
    )

    col1, col2 = st.columns(2)

    with col1:

        normal_avg = df[
            df["Class"] == 0
        ]["Amount"].mean()

        st.metric(
            "Average Normal Transaction",
            f"${normal_avg:.2f}"
        )

    with col2:

        fraud_avg = df[
            df["Class"] == 1
        ]["Amount"].mean()

        st.metric(
            "Average Fraud Transaction",
            f"${fraud_avg:.2f}"
        )

    st.markdown("---")

    st.info(
        "The system first uses K-Means clustering to identify "
        "transaction behavior patterns and then uses Random "
        "Forest for supervised fraud classification."
    )


# ============================================================
# FRAUD DETECTION
# ============================================================

elif page == "🔍 Fraud Detection":

    st.title(
        "🔍 Fraud Detection"
    )

    st.markdown(
        """
        Upload a CSV file containing transaction information.
        The system will apply **K-Means clustering** and
        **Random Forest classification** to identify potentially
        fraudulent transactions.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # REQUIRED FEATURES
    # --------------------------------------------------------

    expected_features = [
        "Time"
    ]

    expected_features += [
        f"V{i}"
        for i in range(1, 29)
    ]

    expected_features += [
        "Amount"
    ]

    st.subheader(
        "📁 Upload Transaction Data"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(
                uploaded_file
            )
            # Limit the number of rows processed at once
            MAX_ROWS = 10000
            if len(uploaded_data) > MAX_ROWS:
               st.warning(
                   f"The uploaded file contains {len(uploaded_data):,} rows. "
                   f"Only the first {MAX_ROWS:,} rows will be processed "
                   "to keep the application responsive."
                )
            uploaded_data = uploaded_data.head(MAX_ROWS).copy()
            st.success(
                f"File uploaded successfully. "
                f"{len(uploaded_data):,} transactions found."
            )

            # ------------------------------------------------
            # CHECK REQUIRED COLUMNS
            # ------------------------------------------------

            missing_columns = [
                column
                for column in expected_features
                if column not in uploaded_data.columns
            ]

            if missing_columns:

                st.error(
                    "The uploaded CSV is missing required columns:"
                )

                st.write(
                    missing_columns
                )

                st.info(
                    "Required columns are: "
                    + ", ".join(expected_features)
                )

                st.stop()

            # ------------------------------------------------
            # KEEP ONLY MODEL FEATURES
            # ------------------------------------------------

            prediction_input = uploaded_data[
                expected_features
            ].copy()

            # ------------------------------------------------
            # SCALE TIME AND AMOUNT
            # ------------------------------------------------

            prediction_input[
                ["Time", "Amount"]
            ] = scaler.transform(
                prediction_input[
                    ["Time", "Amount"]
                ]
            )

            # ------------------------------------------------
            # K-MEANS CLUSTERING
            # ------------------------------------------------

            clusters = kmeans.predict(
                prediction_input
            )

            cluster_distances = kmeans.transform(
                prediction_input
            ).min(axis=1)

            # ------------------------------------------------
            # ADD K-MEANS FEATURES
            # ------------------------------------------------

            prediction_input[
                "Cluster"
            ] = clusters

            prediction_input[
                "Cluster_Distance"
            ] = cluster_distances

            # ------------------------------------------------
            # RANDOM FOREST PREDICTION
            # ------------------------------------------------

            predictions = model.predict(
                prediction_input
            )

            probabilities = model.predict_proba(
                prediction_input
            )[:, 1]

            # ------------------------------------------------
            # CREATE RESULTS
            # ------------------------------------------------

            results = uploaded_data.copy()

            results[
                "Prediction"
            ] = np.where(
                predictions == 1,
                "Fraud",
                "Normal"
            )

            results[
                "Fraud Probability (%)"
            ] = (
                probabilities * 100
            ).round(2)

            results[
                "K-Means Cluster"
            ] = clusters

            results[
                "Cluster Distance"
            ] = cluster_distances.round(4)

            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            def get_risk_level(probability):

                if probability >= 80:

                    return "High Risk"

                elif probability >= 50:

                    return "Medium Risk"

                else:

                    return "Low Risk"

            results[
                "Risk Level"
            ] = [
                get_risk_level(
                    probability * 100
                )
                for probability in probabilities
            ]

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "📊 Prediction Summary"
            )

            fraud_count = int(
                np.sum(predictions == 1)
            )

            normal_count = int(
                np.sum(predictions == 0)
            )

            high_risk_count = int(
                np.sum(
                    probabilities >= 0.80
                )
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Total",
                    len(results)
                )

            with col2:

                st.metric(
                    "Normal",
                    normal_count
                )

            with col3:

                st.metric(
                    "Fraud",
                    fraud_count
                )

            with col4:

                st.metric(
                    "High Risk",
                    high_risk_count
                )

            # ------------------------------------------------
            # RESULTS TABLE
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "🔎 Prediction Results"
            )

            display_columns = [
                "Prediction",
                "Fraud Probability (%)",
                "Risk Level",
                "K-Means Cluster",
                "Cluster Distance"
            ]

            # Add Amount if available
            if "Amount" in results.columns:

                display_columns.insert(
                    0,
                    "Amount"
                )

            st.dataframe(
                results[display_columns],
                width="stretch"
            )

            # ------------------------------------------------
            # CLUSTER DISTRIBUTION
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "🔵 K-Means Cluster Distribution"
            )

            cluster_counts = (
                results[
                    "K-Means Cluster"
                ]
                .value_counts()
                .sort_index()
            )

            cluster_chart = pd.DataFrame(
                {
                    "Cluster": cluster_counts.index.astype(str),
                    "Transactions": cluster_counts.values
                }
            )

            st.bar_chart(
                cluster_chart.set_index(
                    "Cluster"
                )
            )

            # ------------------------------------------------
            # FRAUD PROBABILITY DISTRIBUTION
            # ------------------------------------------------

            st.subheader(
                "📈 Fraud Probability Distribution"
            )

            probability_chart = pd.DataFrame(
                {
                    "Fraud Probability": probabilities
                }
            )

            st.bar_chart(
                probability_chart
            )

            # ------------------------------------------------
            # DOWNLOAD RESULTS
            # ------------------------------------------------

            st.markdown("---")

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8")

            st.subheader("🔍 Prediction Results")

            st.dataframe(
            results,
            width="stretch",
            height=400
            )

        except Exception as e:

            st.error(
                "An error occurred while processing the file."
            )

            st.exception(e)

    else:

        st.info(
            "Upload a CSV file containing Time, V1-V28 and Amount."
        )

        st.markdown(
            """
            **Required columns:**

            `Time, V1, V2, V3, ... V28, Amount`
            """
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title(
        "📊 Transaction Analytics"
    )

    # --------------------------------------------------------
    # TRANSACTION AMOUNT
    # --------------------------------------------------------

    st.subheader(
        "💰 Transaction Amount Distribution"
    )

    amount_data = df[
        "Amount"
    ].clip(
        upper=df["Amount"].quantile(0.99)
    )

    st.line_chart(
        amount_data.head(500)
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📌 Transaction Class Distribution"
    )

    class_distribution = pd.DataFrame(
        {
            "Transaction Type": [
                "Normal",
                "Fraud"
            ],
            "Count": [
                normal_transactions,
                fraud_transactions
            ]
        }
    )

    st.bar_chart(
        class_distribution.set_index(
            "Transaction Type"
        )
    )

    # --------------------------------------------------------
    # FRAUD STATISTICS
    # --------------------------------------------------------

    st.subheader(
        "🚨 Fraud Statistics"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Fraud Transactions",
            f"{fraud_transactions:,}"
        )

    with col2:

        st.metric(
            "Fraud Percentage",
            f"{fraud_rate:.4f}%"
        )

    with col3:

        total_fraud_amount = df[
            df["Class"] == 1
        ]["Amount"].sum()

        st.metric(
            "Total Fraud Amount",
            f"${total_fraud_amount:,.2f}"
        )

    # --------------------------------------------------------
    # NORMAL VS FRAUD AMOUNT
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "💵 Normal vs Fraud Transaction Amount"
    )

    amount_comparison = pd.DataFrame(
        {
            "Average Amount": [
                df[
                    df["Class"] == 0
                ]["Amount"].mean(),

                df[
                    df["Class"] == 1
                ]["Amount"].mean()
            ]
        },
        index=[
            "Normal",
            "Fraud"
        ]
    )

    st.bar_chart(
        amount_comparison
    )

    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📋 Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.title(
        "🤖 Model Performance"
    )

    st.markdown(
        """
        ### Machine Learning Architecture

        **K-Means Clustering → Cluster Features → Random Forest**
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "⚙️ Model Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Primary Algorithm:**"
        )

        st.write(
            "Random Forest Classifier"
        )

        st.write(
            "**Clustering Algorithm:**"
        )

        st.write(
            "K-Means Clustering"
        )

        st.write(
            "**Number of Clusters:**"
        )

        st.write(
            metadata.get(
                "kmeans_clusters",
                3
            )
        )

        st.write(
            "**Random Forest Trees:**"
        )

        st.write(
            metadata.get(
                "random_forest_estimators",
                100
            )
        )

    with col2:

        st.write(
            "**Class Weight:**"
        )

        st.write(
            metadata.get(
                "class_weight",
                "balanced"
            )
        )

        st.write(
            "**Random State:**"
        )

        st.write(
            metadata.get(
                "random_state",
                42
            )
        )

        st.write(
            "**Training Samples:**"
        )

        st.write(
            f"{metadata.get('training_samples', 0):,}"
        )

        st.write(
            "**Testing Samples:**"
        )

        st.write(
            f"{metadata.get('testing_samples', 0):,}"
        )

    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📈 Performance Metrics"
    )

    roc_auc = metadata.get(
        "roc_auc",
        None
    )

    pr_auc = metadata.get(
        "pr_auc",
        None
    )

    col1, col2 = st.columns(2)

    with col1:

        if roc_auc is not None:

            st.metric(
                "ROC-AUC",
                f"{roc_auc:.4f}"
            )

    with col2:

        if pr_auc is not None:

            st.metric(
                "PR-AUC",
                f"{pr_auc:.4f}"
            )

    # --------------------------------------------------------
    # EVALUATION IMAGES
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📊 Evaluation Results"
    )

    confusion_matrix_path = os.path.join(
        EVALUATION_DIR,
        "confusion_matrix.png"
    )

    roc_curve_path = os.path.join(
        EVALUATION_DIR,
        "roc_curve.png"
    )

    pr_curve_path = os.path.join(
        EVALUATION_DIR,
        "precision_recall_curve.png"
    )

    if os.path.exists(
        confusion_matrix_path
    ):

        st.image(
            confusion_matrix_path,
            caption="Confusion Matrix",
            use_container_width=True
        )

    if os.path.exists(
        roc_curve_path
    ):

        st.image(
            roc_curve_path,
            caption="ROC Curve",
            use_container_width=True
        )

    if os.path.exists(
        pr_curve_path
    ):

        st.image(
            pr_curve_path,
            caption="Precision-Recall Curve",
            use_container_width=True
        )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.title(
        "ℹ️ About the Project"
    )

    st.markdown(
        """
        ## 💳 AI-Powered Credit Card Fraud Detection

        This project uses a combination of **unsupervised and
        supervised machine learning techniques** to detect
        potentially fraudulent credit card transactions.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # TECHNOLOGIES
    # --------------------------------------------------------

    st.subheader(
        "🛠️ Technologies Used"
    )

    technologies = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Pandas",
                "NumPy",
                "Scikit-learn",
                "K-Means",
                "Random Forest",
                "Streamlit",
                "Joblib"
            ],
            "Purpose": [
                "Programming Language",
                "Data Processing",
                "Numerical Computation",
                "Machine Learning",
                "Transaction Clustering",
                "Fraud Classification",
                "Web Application",
                "Model Serialization"
            ]
        }
    )

    st.table(
        technologies
    )

    # --------------------------------------------------------
    # PROJECT WORKFLOW
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🔄 Project Workflow"
    )

    st.markdown(
        """
        **1. Data Collection**

        Credit card transaction data is loaded and analyzed.

        **2. Data Preprocessing**

        Missing values, duplicate records and feature scaling
        are handled.

        **3. K-Means Clustering**

        Transactions are grouped into behavioral clusters.

        **4. Cluster-Based Features**

        The cluster number and distance from the nearest cluster
        center are generated as additional features.

        **5. Random Forest Classification**

        The original transaction features together with the
        K-Means features are used to classify transactions as
        normal or fraudulent.

        **6. Fraud Probability**

        The Random Forest model produces a probability score
        indicating the likelihood of fraud.

        **7. Risk Classification**

        Transactions are categorized into:

        - 🟢 Low Risk
        - 🟡 Medium Risk
        - 🔴 High Risk

        **8. Streamlit Deployment**

        Users can upload transaction CSV files and receive
        real-time predictions.
        """
    )

    # --------------------------------------------------------
    # ML ARCHITECTURE
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🧠 Machine Learning Architecture"
    )

    st.code(
        """
Credit Card Transaction
          ↓
Data Preprocessing
          ↓
StandardScaler
          ↓
K-Means Clustering
          ↓
Cluster + Cluster Distance
          ↓
Random Forest Classifier
          ↓
Fraud Probability
          ↓
Risk Classification
        """,
        language="text"
    )

    # --------------------------------------------------------
    # K-MEANS EXPLANATION
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🔵 Why K-Means?"
    )

    st.write(
        """
        K-Means is used to identify groups of transactions with
        similar behavioral characteristics. The resulting cluster
        information provides additional information to the Random
        Forest classifier.
        """
    )

    # --------------------------------------------------------
    # RANDOM FOREST EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "🌳 Why Random Forest?"
    )

    st.write(
        """
        Random Forest is a supervised machine learning algorithm
        that combines multiple decision trees. It is used to learn
        the difference between normal and fraudulent transactions.
        """
    )

    # --------------------------------------------------------
    # PROJECT OBJECTIVE
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🎯 Project Objective"
    )

    st.write(
        """
        The objective of this project is to develop an intelligent
        fraud detection system that combines transaction behavior
        analysis through K-Means clustering with supervised fraud
        classification using Random Forest.
        """
    )

    st.success(
        "K-Means + Random Forest fraud detection system is ready."
    )