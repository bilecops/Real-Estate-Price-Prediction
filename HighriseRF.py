import streamlit as st
from joblib import load
import numpy as np
import math
from datetime import datetime

# Load the trained Random Forest model
model_path = "RandomForest_Highrise.joblib"  # Adjust the path as needed
model = load(model_path)

# Title of the app
st.title("Random Forest Prediction for Highrise Prices")

# Class descriptions for each feature
st.markdown("""
**Feature Descriptions:**
1. **Log Land/Parcel Area:** Logarithmic transformation of the land or parcel area. 
   (Log of the original square footage.)
2. **Log Main Floor Area:** Logarithmic transformation of the main floor area.
   (Log of the original square footage.)
3. **Transaction Date (Ordinal):** The date of the property transaction converted to an ordinal date (integer form).
4. **Property Type:** Categorical variable indicating the type of property (e.g., 'Condominium/Apartment', 'Flat', 'Low-Cost Flat', 'Town House').
5. **Mukim:** A geographical region code representing the location of the property (e.g., 'Kuala Lumpur Town Centre', 'Mukim Ampang', etc.).
6. **Tenure:** Indicates the property tenure ('Freehold' or 'Leasehold').
""")

# Converter for the log-transformed features
def log_to_original(log_value):
    return math.exp(log_value)  # Convert log to original scale using exp

# Converter for the log-transformed features
def log_to_original(log_value):
    return math.exp(log_value)  # Convert log to original scale using exp

# Input features with class descriptions
feature1 = st.number_input("Log Land/Parcel Area", value=7.5, step=0.1)
original_feature1 = log_to_original(feature1) # Show original value (square footage) for user information
st.write(f"Original Land/Parcel Area: {original_feature1:,.2f} square feet") # Convert log values back to original (square footage)

feature2 = st.number_input("Log Main Floor Area", value=8.0, step=0.1)
original_feature2 = log_to_original(feature2) # Show original value (square footage) for user information
st.write(f"Original Main Floor Area: {original_feature2:,.2f} square feet") # Convert log values back to original (square footage)

# Convert transaction date to ordinal
feature3 = st.date_input("Transaction Date", value=datetime(2020, 1, 1))

# Function to convert a date to its ordinal form
def date_to_ordinal(date):
    return date.toordinal()

ordinal_feature3 = date_to_ordinal(feature3)

# Show original date
st.write(f"Transaction Date (Ordinal): {ordinal_feature3}")

# Property Type (Categorical feature with encoding)
property_types = ['Condominium/Apartment', 'Flat', 'Low-Cost Flat', 'Town House']
property_type_mapping = {pt: idx for idx, pt in enumerate(property_types)}  # Create encoding map
feature4 = st.selectbox("Property Type", options=property_types, index=0)
encoded_feature4 = property_type_mapping[feature4]  # Encode selected value

# Mukim (Geographical region, Categorical feature with encoding)
mukims = ['Kuala Lumpur Town Centre', 'Mukim Ampang', 'Mukim Batu', 'Mukim Cheras',
          'Mukim Kuala Lumpur', 'Mukim Petaling', 'Mukim Setapak', 'Mukim Ulu Kelang']
mukim_mapping = {mukim: idx for idx, mukim in enumerate(mukims)}  # Create encoding map
feature5 = st.selectbox("Mukim", options=mukims, index=0)
encoded_feature5 = mukim_mapping[feature5]  # Encode selected value

# Tenure (Categorical feature with encoding)
tenures = ['Freehold', 'Leasehold']
tenure_mapping = {tenure: idx for idx, tenure in enumerate(tenures)}  # Create encoding map
feature6 = st.selectbox("Tenure", options=tenures, index=0)
encoded_feature6 = tenure_mapping[feature6]  # Encode selected value

# Predict button
if st.button("Predict Price"):
    # Prepare input features as a NumPy array (including encoded categorical features)
    input_features = np.array([[feature1, feature2, ordinal_feature3, encoded_feature4, encoded_feature5, encoded_feature6]])

    # Perform prediction
    predicted_price = model.predict(input_features)[0]

    # Display the predicted price
    st.subheader(f"Predicted Price (RM): {predicted_price:,.2f}")

