import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load model and preprocessors
model  = tf.keras.models.load_model('model.h5')
scaler = pickle.load(open('scaller.pkl', 'rb'))
le     = pickle.load(open('label_encoder.pkl', 'rb'))
oh     = pickle.load(open('oh.pkl', 'rb'))

# ── UI ──────────────────────────────────────────────
st.title('Customer Churn Prediction 🏦')
st.markdown('Fill in the customer details below to predict churn probability.')

# ── Input Fields ─────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    geography        = st.selectbox('Geography',         oh.categories_[0])
    gender           = st.selectbox('Gender',            le.classes_)
    age              = st.slider('Age',                  min_value=18, max_value=92, value=35)
    tenure           = st.slider('Tenure',               min_value=0,  max_value=10, value=5)
    num_of_products  = st.slider('Number of Products',   min_value=1,  max_value=4,  value=1)

with col2:
    credit_score     = st.number_input('Credit Score',     min_value=300,  max_value=850,   value=600)
    balance          = st.number_input('Balance',          min_value=0.0,                   value=50000.0)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0,                   value=50000.0)
    has_cr_card      = st.selectbox('Has Credit Card',     [0, 1])
    is_active_member = st.selectbox('Is Active Member',    [0, 1])

# ── Predict Button ───────────────────────────────────
if st.button('🔍 Predict Churn', use_container_width=True):

    # Prepare input DataFrame
    input_data = pd.DataFrame({
        'CreditScore':     [credit_score],
        'Gender':          [le.transform([gender])[0]],
        'Age':             [age],
        'Tenure':          [tenure],
        'Balance':         [balance],
        'NumOfProducts':   [num_of_products],
        'HasCrCard':       [has_cr_card],
        'IsActiveMember':  [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded    = oh.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=oh.get_feature_names_out(['Geography']))

    # Combine input + geography encoded
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction      = model.predict(input_scaled)
    prediction_proba = float(prediction[0][0])

    # ── Show Result ──────────────────────────────────
    st.subheader('~ Prediction Result ~')

    if prediction_proba > 0.5:
        st.error('Customer is likely to CHURN')
    else:
        st.success('Customer is likely to STAY')

    st.metric(label='Churn Probability', value=f'{prediction_proba * 100:.2f}%')
    st.progress(prediction_proba)