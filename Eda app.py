import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="📊 Professional EDA App", layout="wide")

st.title("📊 Professional EDA Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # ---------------------- Missing Values ----------------------
    st.subheader("🩹 Missing Values Overview")
    missing_info = df.isnull().sum()
    st.write(missing_info[missing_info > 0])  # show only columns with missing values

    # Fill missing values (simple imputation)
    df = df.fillna(df.median(numeric_only=True))
    st.success("✅ Missing values handled (numeric filled with median, categorical unchanged).")

    # ---------------------- Dataset Overview ----------------------
    st.subheader("📄 Dataset Overview")
    st.write(df.head())
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # ---------------------- Numeric & Categorical ----------------------
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.write(f"📊 Numeric Columns: {num_cols}")
    st.write(f"🔠 Categorical Columns: {cat_cols}")

    # ---------------------- Correlation Heatmap ----------------------
    st.subheader("🔥 Correlation Heatmap (Numeric Only)")
    selected_cols = st.multiselect("Select numeric columns for heatmap:", num_cols, default=num_cols)
    if selected_cols:
        corr = df[selected_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(fig)

    # ---------------------- Outlier Detection ----------------------
    st.subheader("📦 Outlier Detection (Boxplot)")
    col_outlier = st.selectbox("Select column for outlier detection:", num_cols)
    if col_outlier:
        fig = px.box(df, y=col_outlier, points="all",
                     title=f"Outlier Detection for {col_outlier}",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------- Violin Plot ----------------------
    st.subheader("🎻 Violin Plot")
    if num_cols and cat_cols:
        col_num = st.selectbox("Select numeric column:", num_cols)
        col_cat = st.selectbox("Select categorical column:", cat_cols)
        fig = px.violin(df, x=col_cat, y=col_num, box=True, points="all",
                        color=col_cat, 
                        color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------- Pairplot ----------------------
    if len(num_cols) > 1:
        st.subheader("🔗 Pairplot (Sampled for Speed)")
        sample_df = df.sample(min(200, len(df)), random_state=42)
        fig = sns.pairplot(sample_df[num_cols], diag_kind="kde", palette="husl")
        st.pyplot(fig)

    # ---------------------- Distribution Plot ----------------------
    st.subheader("📈 Distribution Plot")
    dist_col = st.selectbox("Select a column for distribution:", num_cols)
    if dist_col:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[dist_col], kde=True, color="purple")
        st.pyplot(fig)

    # Final Note
    st.info("✅ EDA completed. You can extend this app with ML models, feature engineering, etc.")

else:
    st.warning("📂 Please upload a CSV file to start the EDA.")
