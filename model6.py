import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load pre-trained models
try:
    model1 = pickle.load(open('best_model_non1.pkl', 'rb'))  # Ensure the model file path is correct
    model2 = pickle.load(open('best_model_int.pkl', 'rb'))
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")

# Load the CSV file, retrieve feature names and ranges
df = pd.read_csv('Dataset-Non.csv')  # Load the uploaded CSV file

# Get feature columns (assuming the last column is the target column)
input_columns = df.columns[:-1]  # Get all features excluding the target column
target_column = df.columns[-1]   # Assume the last column is the target column (Target)

# Get the min and max values of features
feature_ranges = df[input_columns].describe().transpose()[['min', 'max']]  # Get min and max for each feature

# Create Streamlit interface
st.title("Model Prediction Application")

# Display feature ranges
st.write("The valid range for each input feature is as follows:")
st.dataframe(feature_ranges)

# Create input boxes in a 3-column layout
columns = st.columns(3)  # Create three columns

feature_values = []
label_encoder = LabelEncoder()

# Loop through the features and display sliders for each feature
for idx, feature in enumerate(input_columns):
    min_val = feature_ranges.loc[feature, 'min']
    max_val = feature_ranges.loc[feature, 'max']

    with columns[idx % 3]:  # Distribute input fields across columns
        # For categorical features, use LabelEncoder to convert to integers
        if df[feature].dtype == 'object':  # Categorical data
            st.write(f"{feature} is a categorical feature and will be encoded as integers.")
            df[feature] = label_encoder.fit_transform(df[feature])  # Convert to integers
            user_input = st.selectbox(f"Select a value for {feature}", options=df[feature].unique())
        
        else:  # For numeric features
            if feature == 'Nationality':  # For Nationality, restrict input to values between 1 and 21
                user_input = st.slider(f"Select the value for {feature} (min: 1, max: 21)", 
                                       min_value=1, max_value=21, value=1, step=1)
                st.write(f"Current input value for {feature} is: {user_input}")
            
            elif feature in ['Unemployment', 'Inflation']:  # Keep one decimal place
                user_input = st.slider(f"Select the value for {feature} (min: {min_val}, max: {max_val})", 
                                       min_value=float(min_val), max_value=float(max_val), value=float(min_val), step=0.1)
                st.write(f"Current input value for {feature} is: {round(user_input, 1):.1f}")
            
            elif feature == 'GDP':  # Keep two decimal places
                user_input = st.slider(f"Select the value for {feature} (min: {min_val}, max: {max_val})", 
                                       min_value=float(min_val), max_value=float(max_val), value=float(min_val), step=0.01)
                st.write(f"Current input value for {feature} is: {round(user_input, 2):.2f}")
            
            else:  # Other numeric features
                user_input = st.slider(f"Select the value for {feature} (min: {min_val}, max: {max_val})", 
                                       min_value=int(min_val), max_value=int(max_val), value=int(min_val), step=1)
        
    feature_values.append(user_input)

if st.button("Predict"):
    try:
        # Load the saved Scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)  # Load the Scaler object

        # Use the Scaler for standardization
        feature_values_scaled = scaler.transform([feature_values])  # Apply the same scaler for transformation

        # Check if Nationality feature is in the valid range
        nationality_index = df.columns.get_loc('Nationality')  # Get the index of the Nationality column
        nationality_value = feature_values[nationality_index]  # Get the value entered for Nationality

        if 2 <= nationality_value <= 21:  # If the value of Nationality is between 2 and 21
            # Use the best_model_int.pkl for prediction
            prediction = model2.predict(feature_values_scaled)  # Use the integer model for prediction
            st.write("Prediction result using best_model_int.pkl:", prediction)

            if prediction[0] == 0:
                st.success(f"Predicted target value: Not Graduated")
            elif prediction[0] == 1:
                st.success(f"Predicted target value: Graduated")
            else:
                st.warning(f"Prediction result is out of expected range: {prediction[0]}")
        else:
            # Use the best_model_non1.pkl for prediction
            prediction = model1.predict(feature_values_scaled)  # Use the non-integer model for prediction
            st.write("Prediction result using best_model_non1.pkl:", prediction)

            if prediction[0] == 0:
                st.success(f"Predicted target value: Not Graduated")
            elif prediction[0] == 1:
                st.success(f"Predicted target value: Graduated")
            else:
                st.warning(f"Prediction result is out of expected range: {prediction[0]}")

    except ValueError as e:
        st.error(f"Invalid input value: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")