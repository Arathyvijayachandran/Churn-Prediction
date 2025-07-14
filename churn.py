import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import shap

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction with XGBoost and Clustering")

# Upload dataset
uploaded_file = st.file_uploader("Upload your customer dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Convert target column
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    st.subheader("Exploratory Data Analysis")

    # Plot churn rate by categorical features
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        churn_rate = df.groupby(col)['Churn'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots()
        churn_rate.plot(kind='bar', ax=ax)
        ax.set_title(f"Churn Rate by {col}")
        ax.set_ylabel("Churn Rate")
        st.pyplot(fig)

    # KMeans clustering
    st.subheader("Customer Segmentation with KMeans")

    cluster_data = df[['tenure', 'MonthlyCharges']].dropna()
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(cluster_scaled)

    fig, ax = plt.subplots()
    scatter = ax.scatter(df['tenure'], df['MonthlyCharges'], c=df['Cluster'], cmap='viridis')
    ax.set_xlabel("Tenure")
    ax.set_ylabel("Monthly Charges")
    ax.set_title("KMeans Clustering")
    st.pyplot(fig)

    # Feature encoding
    st.subheader("Feature Engineering and Encoding")

    binary_cols = [col for col in df.select_dtypes(include='object').columns if df[col].nunique() == 2]
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, drop_first=True)

    # Train/Test split
    st.subheader("Model Training with XGBoost")

    target = 'Churn'
    features = df.drop(columns=['Churn'], errors='ignore').columns
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model Accuracy: {acc:.2f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Feature importance
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax)
    st.pyplot(fig)

    # Churn probability scoring
    st.subheader("Churn Probability Scoring")

    churn_probs = model.predict_proba(X_test)[:, 1]
    results_df = X_test.copy()
    results_df['Churn_Prob'] = churn_probs
    results_df['Actual_Churn'] = y_test.values

    st.write("Top 5 Customers at Highest Risk of Churn:")
    st.dataframe(results_df.sort_values(by='Churn_Prob', ascending=False).head(5))

    # Distribution Plot
    st.subheader("Churn Probability Distribution")
    fig, ax = plt.subplots()
    sns.histplot(results_df['Churn_Prob'], bins=20, kde=True, ax=ax)
    ax.set_title("Churn Probability Distribution")
    st.pyplot(fig)
