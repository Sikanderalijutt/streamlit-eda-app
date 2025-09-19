# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Title
st.title("📊 Professional Exploratory Data Analysis (EDA) App")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Dataset Preview
    st.subheader("🔎 Dataset Preview")
    st.write(df.head())

    # Dataset Info
    st.subheader("📌 Dataset Info")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("Columns:", list(df.columns))

    # Missing values
    st.subheader("❓ Missing Values")
    st.write(df.isnull().sum())

    # Handling missing values
    st.subheader("🛠️ Handle Missing Values")
    missing_option = st.radio(
        "Choose a method to handle missing values:",
        ("Do nothing", "Drop rows", "Drop columns", "Fill with mean/median/mode")
    )

    if missing_option == "Drop rows":
        df = df.dropna()
        st.info("Dropped rows with missing values.")
    elif missing_option == "Drop columns":
        df = df.dropna(axis=1)
        st.info("Dropped columns with missing values.")
    elif missing_option == "Fill with mean/median/mode":
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ["int64", "float64"]:
                    fill_method = st.selectbox(
                        f"Choose fill method for numeric column {col}",
                        ("Mean", "Median")
                    )
                    if fill_method == "Mean":
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        st.info("Missing values filled.")

    # Summary statistics
    st.subheader("📊 Summary Statistics")
    st.write(df.describe(include="all"))

    # Separate numeric & categorical
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Correlation heatmap (numeric only)
    st.subheader("🔥 Correlation Heatmap (Numeric Only)")
    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

    # Outlier detection
    st.subheader("🚨 Outlier Detection")
    col_outlier = st.selectbox("Select a column for outlier detection", num_cols)
    if col_outlier:
        fig = px.box(df, y=col_outlier, points="all",
                     title=f"Outlier Detection for {col_outlier}")
        st.plotly_chart(fig, use_container_width=True)

    # Distribution plots
    st.subheader("📈 Distribution of Numeric Features")
    col_dist = st.selectbox("Select a numeric column for distribution", num_cols)
    if col_dist:
        fig, ax = plt.subplots()
        sns.histplot(df[col_dist], kde=True, bins=20, ax=ax)
        ax.set_title(f"Distribution of {col_dist}")
        st.pyplot(fig)

    # Categorical vs Numeric analysis
    st.subheader("📊 Categorical vs Numeric Analysis")
    if cat_cols and num_cols:
        cat_col = st.selectbox("Select categorical column", cat_cols)
        num_col = st.selectbox("Select numeric column", num_cols)

        fig, ax = plt.subplots()
        sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
        ax.set_title(f"{num_col} by {cat_col}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.violinplot(x=df[cat_col], y=df[num_col], ax=ax)
        ax.set_title(f"Violin Plot of {num_col} by {cat_col}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Scatter plot
    st.subheader("🔗 Scatter Plot (Numeric vs Numeric)")
    if len(num_cols) > 1:
        x_axis = st.selectbox("Select X-axis", num_cols, index=0)
        y_axis = st.selectbox("Select Y-axis", num_cols, index=1)

        fig = px.scatter(df, x=x_axis, y=y_axis,
                         color=df[cat_cols[0]] if cat_cols else None,
                         title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    # Pairplot
    st.subheader("📊 Pairplot (Sampled Numeric Columns)")
    if len(num_cols) > 1:
        sample_df = df.sample(min(200, len(df)), random_state=42)
        fig = sns.pairplot(sample_df[num_cols[:4]])  # limit to 4 cols
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for pairplot.")

    # Final note
    st.success("✅ EDA completed. You can now extend this app with ML models, feature engineering, etc.")
else:
    st.warning("📂 Please upload a CSV file to start the EDA.")
