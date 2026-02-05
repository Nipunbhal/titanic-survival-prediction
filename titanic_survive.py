import streamlit as st
import pandas as pd
import joblib

# --- Load model ---
model = joblib.load("titanic_logreg_model.pkl")

# --- App Title ---
st.title("🛳️ Titanic Survival Prediction App")
st.write("Enter passenger details below to predict survival chance.")

# --- User Inputs ---
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 29)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
fare = st.slider("Passenger Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
deck = st.selectbox("Deck", ["A", "B", "C", "D", "E", "F", "G", "U"])
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# --- Create input dataframe ---
input_df = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked],
    'Title': [title],
    'Deck': [deck],
    'FamilySize': [family_size],
    'IsAlone': [is_alone]
})

# --- Prediction ---
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"🎉 The passenger would likely SURVIVE! (Prob: {probability:.2f})")
    else:
        st.error(f"💀 The passenger would NOT survive. (Prob: {probability:.2f})")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed by Nipun Bhal • Machine Learning Mini Project • Titanic Dataset")
