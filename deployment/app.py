# deployment/app.py

import streamlit as st
from eda import run_eda
from prediction import run_prediction

# Page configuration
st.set_page_config(page_title="Credit Card Default Prediction", layout="wide")

def main():
    st.title("💳 Credit Card Default Prediction (Age 25–35)")
    st.sidebar.title("Navigation")

    menu = ["Project Overview", "Exploratory Data Analysis", "Prediction"]
    choice = st.sidebar.selectbox("Go to", menu)

    if choice == "Project Overview":
        st.subheader("📌 Project Description")
        st.write("""
        This project aims to predict whether a credit card customer (aged 25–35) will default based on key financial and demographic features.
        
        **Objective**:
        - Predict default status (binary classification)
        - Handle imbalanced data
        - Evaluate various ML models (KNN was selected as best for deployment)
        - Deploy model with Streamlit app

        **Model**: K-Nearest Neighbors (KNN)  
        **Data**: Filtered from Kaggle dataset, focusing only on users aged 25–35  
        **Target**: `TARGET` (1 = Default, 0 = Not Default)

        """)
    
    elif choice == "Exploratory Data Analysis":
        run_eda()
    
    elif choice == "Prediction":
        run_prediction()

if __name__ == '__main__':
    main()
