# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("📊 Exploratory Data Analysis (EDA) App")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show dataset preview
    st.subheader("🔎 Dataset Preview")
    st.write(df.head())

    # Show basic info
    st.subheader("📌 Dataset Info")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("Columns:", list(df.columns))

    # Missing values
    st.subheader("❓ Missing Values")
    st.write(df.isnull().sum())

    # Summary statistics
    st.subheader("📊 Summary Statistics")
    st.write(df.describe(include="all"))

    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap (Numeric Columns)")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns available for correlation heatmap.")

    # Column-wise analysis
    st.subheader("📈 Column-wise Analysis")
    column = st.selectbox("Select a column for analysis", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        st.write(f"Summary of **{column}**:")
        st.write(df[column].describe())

        # Histogram
        fig, ax = plt.subplots()
        sns.histplot(df[column], bins=20, kde=True, ax=ax)
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)

        # Boxplot
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], ax=ax)
        ax.set_title(f"Boxplot of {column}")
        st.pyplot(fig)

    else:
        st.write(f"Value counts of **{column}**:")
        st.write(df[column].value_counts())

        # Bar chart
        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Bar Chart of {column}")
        st.pyplot(fig)

    # Pairplot
    st.subheader("📊 Pairplot (Sample of Numeric Columns)")
    if numeric_df.shape[1] > 1:
        sample_cols = numeric_df.columns[:4]  # limit to 4 cols to avoid heavy plots
        fig = sns.pairplot(df[sample_cols])
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for pairplot.")
