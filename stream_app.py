
import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("titanic_model_pipeline.pkl")

st.set_page_config(page_title="Titanic Survival Predictor")

st.title("Titanic Survival Prediction")
st.write("Fill in passenger details to predict survival probability.")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)
fare = st.slider("Fare", 0, 500, 50)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# DataFrame for model
input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

# Predict
if st.button("Predict Survival"):
    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]
    st.subheader("Result")
    if pred == 1:
        st.success(f"Survived (Probability {prob:.2f})")
    else:
        st.error(f" Did Not Survive (Probability {prob:.2f})")  
