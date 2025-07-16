# deployment/eda.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():
    st.subheader("📊 Exploratory Data Analysis")

    df = pd.read_csv("application_data.csv")

    st.write("### Raw Data Sample")
    st.dataframe(df.head())

    st.write("### Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='TARGET', data=df, ax=ax)
    st.pyplot(fig)

    st.write("### Income Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['AMT_INCOME_TOTAL'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='Blues', ax=ax)
    st.pyplot(fig)

if __name__ == '__main__':
    run_eda()