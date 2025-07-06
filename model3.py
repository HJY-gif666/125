import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load pre-trained models
model1 = pickle.load(open('best_model_non.pkl', 'rb'))  # Ensure the model file path is correct
model2 = pickle.load(open('best_model_int.pkl', 'rb'))

# Load the CSV file to get feature names and ranges
df = pd.read_csv('Dataset-Non.csv')  # Load the uploaded CSV file

# Extract feature columns (assuming the last column is the target)
input_columns = df.columns[:-1]  # Get all feature columns except the target
target_column = df.columns[-1]   # Assume the last column is the target column (Target)

# Get feature value ranges (min and max)
feature_ranges = df[input_columns].describe().transpose()[['min', 'max']]  # Get min and max values of each feature

# Create the Streamlit interface
st.title("Model Prediction Application")

# Display feature ranges for user reference
st.write("The valid ranges for the input features are as follows:")
st.dataframe(feature_ranges)

# Input fields for features, with validation to ensure input is within the valid range
feature_values = []
for feature in input_columns:
    min_val = feature_ranges.loc[feature, 'min']
    max_val = feature_ranges.loc[feature, 'max']
    user_input = st.number_input(f"Enter the value for {feature} (min: {min_val}, max: {max_val})", 
                                value=float(min_val), 
                                min_value=float(min_val), 
                                max_value=float(max_val))
    feature_values.append(user_input)

# Predict button
if st.button("Predict"):
    try:
        # Feature standardization (make sure we fit the scaler using the same dataset)
        scaler = StandardScaler()
        feature_values_scaled = scaler.fit_transform([feature_values])

        # Choose the correct model for prediction
        # If you want to select between models, you could add a condition here to choose model1 or model2
        prediction = model1.predict(feature_values_scaled)  # Use the appropriate model as needed

        # Show the prediction result
        st.success(f"Prediction result: {prediction[0]}")

    except ValueError as e:
        st.error(f"Invalid input value(s): {e}")