import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Title
st.title('🔍 Customer Churn Prediction')

st.markdown("Fill in the customer details to predict the chance of churn.")

# User Inputs
# --- Personal Information ---
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', min_value=18, max_value=92, step=1)

# --- Financial Information ---
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, step=1)
balance = st.number_input('Account Balance (in $)', min_value=0.00, format="%d", step=100)
estimated_salary = st.number_input('Estimated Salary (in $)', min_value=0.00, format="%d", step=100)
tenure = st.slider('Tenure (Years with Bank)', min_value=0, max_value=10, step=1)

# --- Product & Usage Information ---
num_of_products = st.slider('Number of Products', min_value=1, max_value=4, step=1)

has_cr_card_display = st.radio('Has Credit Card?', ['Yes', 'No'], horizontal=True)
is_active_member_display = st.radio('Is Active Member?', ['Yes', 'No'], horizontal=True)

# Convert "Yes"/"No" to binary for model input
has_cr_card = 1 if has_cr_card_display == 'Yes' else 0
is_active_member = 1 if is_active_member_display == 'Yes' else 0



# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine all features
final_input = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input
scaled_input = scaler.transform(final_input)

# Prediction
prediction = model.predict(scaled_input)
churn_probability = prediction[0][0]

# Display result
st.markdown(f"### 📊 Churn Probability: `{churn_probability:.2%}`")

if churn_probability > 0.5:
    st.error('⚠️ The customer is **likely to churn**.')
else:
    st.success('✅ The customer is **not likely to churn**.')

# Footer
st.markdown("---")
st.caption("Model powered by TensorFlow | Interface built using Streamlit")
