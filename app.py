
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Salary Prediction App", page_icon="💼")

@st.cache_resource
def load_model():
    return joblib.load("salary_random_forest.joblib")

artifact = load_model()

st.title("💼 Employee Salary Prediction")

age = st.number_input(
    "Age",
    min_value=float(artifact["age_min"]),
    max_value=float(artifact["age_max"]),
    value=float((artifact["age_min"] + artifact["age_max"]) / 2)
)

gender = st.selectbox("Gender", artifact["gender_options"])
education = st.selectbox("Education Level", artifact["education_options"])
job_title = st.selectbox("Job Title", artifact["job_titles"])

experience = st.number_input(
    "Years of Experience",
    min_value=float(artifact["experience_min"]),
    max_value=float(artifact["experience_max"]),
    value=float((artifact["experience_min"] + artifact["experience_max"]) / 2),
    step=0.5
)

if st.button("Predict Salary"):
    input_encoded = pd.DataFrame(
    0.0,
    index=[0],
    columns=artifact["feature_columns"]
)

    input_encoded.loc[0, "Age"] = age
    input_encoded.loc[0, "Years of Experience"] = experience

    input_encoded.loc[0, "Job Title"] = artifact["label_encoder"].transform(
        [job_title]
    )[0]

    gender_column = f"Gender_{gender}"
    if gender_column in input_encoded.columns:
        input_encoded.loc[0, gender_column] = 1

    education_column = f"Education Level_{education}"
    if education_column in input_encoded.columns:
        input_encoded.loc[0, education_column] = 1

    model_input = input_encoded[artifact["selected_features"]]
    prediction = artifact["model"].predict(model_input)[0]

    st.success(f"Predicted Salary: ${prediction:,.2f}")
