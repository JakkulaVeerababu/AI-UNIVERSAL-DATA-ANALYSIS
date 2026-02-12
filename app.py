import streamlit as st
import pandas as pd
import numpy as np
import json
import xmltodict
import os
import requests
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine
from pycaret.clustering import setup as clu_setup, create_model, assign_model
from sklearn.ensemble import IsolationForest

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Universal Data Analysis System",
    layout="wide"
)

st.title("AI Universal Data Analysis System")
st.write("Analyze data from files or MySQL with AI insights.")

# ================= FILE LOADER =================
def load_data(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(file)
    elif ext == ".json":
        return pd.json_normalize(json.load(file))
    elif ext == ".xml":
        return pd.json_normalize(xmltodict.parse(file.read()))
    else:
        raise ValueError("Unsupported file format")

# ================= MYSQL LOADER =================
def load_data_from_mysql():
    engine = create_engine(
        "mysql+pymysql://root:THEFLASH9ZZ%40@localhost/ai_data_db"
    )
    query = "SELECT * FROM expenses"
    df = pd.read_sql(query, engine)
    return df

# ================= CLEAN DATA =================
def clean_data(df):
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    return df.drop_duplicates()

# ================= OLLAMA CALL =================
def ask_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )
    return response.json().get("response", "")

# ================= SIDEBAR =================
st.sidebar.header("Controls")

data_source = st.sidebar.radio(
    "Select data source",
    ["Upload File", "Load from MySQL"]
)

uploaded_file = None
if data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV / Excel / JSON / XML",
        type=["csv", "xlsx", "xls", "json", "xml"]
    )

# ================= MAIN APP =================
if data_source == "Load from MySQL" or uploaded_file:
    try:
        if data_source == "Load from MySQL":
            df = load_data_from_mysql()
            st.success("Data loaded from MySQL")
        else:
            df = clean_data(load_data(uploaded_file))
            st.success("Data loaded from file")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        st.subheader("Data Preview")
        st.dataframe(df.head())

        # ================= EDA =================
        st.subheader("Histogram")
        if numeric_cols:
            col = st.selectbox("Select numeric column", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

        # ================= CLUSTERING =================
        clustered_df = None
        st.subheader("Clustering")

        if st.button("Run Clustering"):
            clu_setup(data=df, session_id=123, verbose=False)
            model = create_model("kmeans")
            clustered_df = assign_model(model)

            st.subheader("Cluster Distribution")
            st.write(clustered_df["Cluster"].value_counts())

        # ================= ANOMALY DETECTION =================
        df_anomaly = None
        st.subheader("Anomaly Detection")

        if numeric_cols and st.button("Detect Anomalies"):
            iso = IsolationForest(contamination=0.05, random_state=42)
            df_anomaly = df.copy()
            df_anomaly["Anomaly"] = iso.fit_predict(df[numeric_cols])
            anomalies = df_anomaly[df_anomaly["Anomaly"] == -1]
            st.write("Anomalies detected:", len(anomalies))
            st.dataframe(anomalies)

        # ================= FULL AI ANALYST =================
        st.subheader("AI Analyst")

        question = st.text_input(
            "Ask anything about the data (insights, patterns, decisions)"
        )

        if question:
            stats = df.describe(include="all").fillna("").to_string()
            sample = df.head(15).to_string()

            cluster_text = ""
            if clustered_df is not None:
                cluster_text = f"""
Cluster distribution:
{clustered_df["Cluster"].value_counts().to_dict()}
"""

            anomaly_text = ""
            if df_anomaly is not None:
                anomaly_text = f"""
Total anomalies:
{(df_anomaly["Anomaly"] == -1).sum()}
"""

            ai_prompt = f"""
You are a senior data scientist.

Columns:
{list(df.columns)}

Statistics:
{stats}

Sample data:
{sample}

{cluster_text}
{anomaly_text}

Question:
{question}

Answer freely with insights and recommendations.
"""
            answer = ask_ollama(ai_prompt)
            if answer:
                st.markdown(answer)

    except Exception as e:
        st.error("Error processing data")
        st.exception(e)

else:
    st.info("Select a data source to begin.")
