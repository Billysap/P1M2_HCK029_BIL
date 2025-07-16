import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def run_eda():
    st.subheader("📊 Exploratory Data Analysis")

    gambar = Image.open('images/creditcarddefaulter.jpg')
    st.write(gambar)

    df = pd.read_csv("application_data.csv")

    st.write("### Raw Data Sample")
    st.dataframe(df.head())

    # Pie chart for target distribution
    st.write("### Target Distribution (Pie Chart)")
    target_counts = df['TARGET'].value_counts()
    labels = ['Not Default (0)', 'Default (1)']
    sizes = target_counts.values
    percentages = [f'{(val / sum(sizes)) * 100:.2f}%' for val in sizes]
    labels_with_counts = [f'{label}: {count:,} ({percent})' for label, count, percent in zip(labels, sizes, percentages)]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels_with_counts, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    ax.axis('equal')
    st.pyplot(fig)

    # Explanation
    st.markdown(
        """
        **Note:**  
        The target classes are imbalanced, with far more non-default (0) cases than default (1).  
        This imbalance can bias machine learning models to favor the majority class.

        To address this, we apply **SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous)** during training.  
        SMOTENC generates synthetic samples for the minority class by considering both categorical and numeric features,  
        improving the model's ability to generalize to unseen default cases.
        """
    )

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='Blues', ax=ax)
    st.pyplot(fig)

if __name__ == '__main__':
    run_eda()
