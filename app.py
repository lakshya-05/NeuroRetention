import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Page configuration
st.set_page_config(
    page_title="NeuroRetention",
    layout="centered"
)

# Load model
model = tf.keras.models.load_model("model.keras")

# Load encoders and scaler
with open("LabelEncoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("OneHotEncoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Title
st.title("NeuroRetention")
st.write("Bank Customer Churn Prediction System")

st.divider()

# User Inputs
credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider(
    "Age",
    18,
    90,
    30
)

tenure = st.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=0.0
)

num_of_products = st.slider(
    "Number of Products",
    1,
    4,
    1
)

has_cr_card = int(
    st.selectbox(
        "Has Credit Card",
        [0, 1]
    )
)

is_active_member = int(
    st.selectbox(
        "Is Active Member",
        [0, 1]
    )
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

# Prediction
if st.button("Predict"):

    # Encode gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Input dataframe
    input_data = pd.DataFrame({
        "CreditScore": [float(credit_score)],
        "Gender": [int(gender_encoded)],
        "Age": [float(age)],
        "Tenure": [float(tenure)],
        "Balance": [float(balance)],
        "NumOfProducts": [float(num_of_products)],
        "HasCrCard": [int(has_cr_card)],
        "IsActiveMember": [int(is_active_member)],
        "EstimatedSalary": [float(estimated_salary)]
    })

    # One-hot encoding for geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    # Merge geography data
    input_data = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    # Match training column order
    expected_columns = scaler.feature_names_in_

    input_data = input_data.reindex(columns=expected_columns)

    # Convert datatype
    input_data = input_data.astype(np.float32)

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    input_data_scaled = np.array(
        input_data_scaled,
        dtype=np.float32
    )

    # Prediction
    prediction = model.predict(
        input_data_scaled,
        verbose=0
    )

    prediction_probability = float(prediction[0][0])

    # Output
    st.divider()

    st.subheader("Prediction Result")

    st.write(
        f"Churn Probability: {prediction_probability:.2%}"
    )

    if prediction_probability > 0.5:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is not likely to churn.")