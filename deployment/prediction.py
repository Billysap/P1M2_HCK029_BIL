# deployment/prediction.py

import streamlit as st
import pandas as pd
import pickle

def run_prediction():
    st.subheader("🔮 Predict Credit Card Default")

    # Load model
    with open("deployment/final_knn_model.pkl", "rb") as file:
        model = pickle.load(file)

    # User input
    gender = st.selectbox("Gender", ['M', 'F'])
    education = st.selectbox("Education", ['Higher education', 'Secondary / secondary special'])
    family = st.selectbox("Family Status", ['Married', 'Single / not married'])
    income_type = st.selectbox("Income Type", ['Working', 'Commercial associate'])
    amt_income = st.number_input("Income Total", min_value=0)
    amt_credit = st.number_input("Credit Amount", min_value=0)
    amt_annuity = st.number_input("Annuity Amount", min_value=0)
    own_car = st.selectbox("Owns Car?", [0, 1])
    own_realty = st.selectbox("Owns Realty?", [0, 1])
    age = st.slider("Age", 25, 35)
    years_employed = st.slider("Years Employed", 0, 20)
    years_id_publish = st.slider("Years Since ID Published", 0, 20)
    cnt_children = st.selectbox("Number of Children", [0, 1, 2, '3+'])

    # Create input DataFrame
    input_data = pd.DataFrame({
        'CODE_GENDER': [gender],
        'NAME_EDUCATION_TYPE': [education],
        'NAME_FAMILY_STATUS': [family],
        'CNT_CHILDREN_GROUPED': [str(cnt_children) if cnt_children != '3+' else '3+'],
        'AGE_YEARS': [age],
        'NAME_INCOME_TYPE_GROUPED': [income_type],
        'YEARS_EMPLOYED': [years_employed],
        'AMT_INCOME_TOTAL': [amt_income],
        'AMT_CREDIT': [amt_credit],
        'AMT_ANNUITY': [amt_annuity],
        'FLAG_OWN_CAR': [own_car],
        'FLAG_OWN_REALTY': [own_realty],
        'YEARS_ID_PUBLISHED': [years_id_publish]
    })

    # Prediction
    if st.button("Predict"):
        result = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        st.success(f"Prediction: {'Default' if result == 1 else 'Not Default'}")
        st.info(f"Probability of Default: {prob:.2%}")

if __name__ == '__main__':
    run_prediction()