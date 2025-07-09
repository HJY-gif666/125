import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load models
model1 = pickle.load(open('best_model_non5.pkl', 'rb'))
model2 = pickle.load(open('best_model_int7.9.pkl', 'rb'))

# Load Scaler object (only once)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Page configuration
st.set_page_config(
    page_title='Evaluating and forecasting undergraduate dropouts',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Page title and extra information
st.markdown("""
    <h1 style='color: darkblue; text-align: center; font-size: 36px;'>Warning forecasts of domestic and international student dropouts</h1>
    <div style='text-align: center; font-size: 18px;'>Developed by Dr. Songbo Wang et al., Hubei University of Technology.</div>
    <div style='text-align: center; font-size: 18px;'>Email: Wangsongbo@hbut.edu.cn</div>
""", unsafe_allow_html=True)

st.sidebar.subheader('Input features for single prediction')
# Use columns to divide into two parts
col1, col2 = st.sidebar.columns(2)

with col1:
    Marital = st.slider('Marital status', 1, 4, 1)
    Mode = st.slider('Application order', 1, 18, 9)
    Order = st.slider("Application order", 1, 5, 2)
    Course = st.slider('Course type', 1, 17, 10)
    Attendance = st.slider('Daytime/evening attendance', 0, 1, 0)
    Qualification = st.slider("Previous qualification", 1, 14, 7)
    Nationality = st.slider("Nationality", 1, 21, 1)
    Mother_Q = st.slider("Mother qualification", 1, 28, 14)
    Father_Q = st.slider('Father qualification', 1, 28, 15)
    Mother_O = st.slider('Mother occupation', 1, 25, 18)
    Father_O = st.slider("Father occupation", 1, 26, 20)
    Displaced = st.slider('Displaced', 0, 1, 0)

with col2:
    Need = st.slider('Educational special need', 0, 1, 0)
    Debtor = st.slider("Debtor", 0, 1, 0)
    Fee = st.slider("Tuition fee", 0, 1, 0)
    Gender = st.slider("Gender", 0, 1, 0)
    Scholarship = st.slider('Scholarship', 0, 1, 0)
    Age = st.slider("Age", 18, 59, 23)
    First = st.slider("1st semester approved course", 0, 18, 9)
    Second = st.slider("2nd semester approved course", 0, 12, 4)
    Unemployment = st.slider("Unemployment rate", 7.6, 16.2, 11.0)
    Inflation = st.slider("Inflation rate", -0.8, 3.7, 1.0)
    GDP = st.slider("GDP", -4.06, 3.51, 1.00)

if st.sidebar.button('Predict'):

    # Input features for prediction
    input_features = [[Marital, Mode, Order, Course, Attendance, Qualification, Nationality,
                       Mother_Q, Father_Q, Mother_O, Father_O, Displaced, Need, Debtor,
                       Fee, Gender, Scholarship, Age, First, Second, Unemployment, Inflation, GDP]]
    
    # Create DataFrame
    new_data = pd.DataFrame(input_features, columns=['Marital', 'Mode', 'Order', 'Course', 'Attendance', 'Qualification',
                                                     'Nationality', 'Mother_Q', 'Father_Q', 'Mother_O', 'Father_O',
                                                     'Displaced', 'Need', 'Debtor', 'Fee', 'Gender', 'Scholarship',
                                                     'Age', 'First', 'Second', 'Unemployment', 'Inflation', 'GDP'])

    # Standardize the input features
    feature_values_scaled = scaler.transform(new_data.values)

    try:
        # Select model based on Nationality value
        if 2 <= Nationality <= 21:  # Use model2 if Nationality value is between 2 and 21
            prediction = model2.predict(feature_values_scaled)
            st.markdown("<h1 style='color: blue; font-size: 30px;'>International students result:</h1>", unsafe_allow_html=True)
        else:  # Otherwise use model1
            prediction = model1.predict(feature_values_scaled)
            st.markdown("<h1 style='color: blue; font-size: 30px;'>Domestic students result:</h1>", unsafe_allow_html=True)

        if prediction[0] == 1:
            st.success("Predicted outcome: Dropout")
        elif prediction[0] == 0:
            st.success("Predicted outcome: Graduated")
            
            # Center the columns
            col1, col2 = st.columns([4, 2])  # Center column is larger

            with col1:
                # Modify to display feature names and values in two columns
                rows = []
                for i in range(0, len(new_data.columns), 2):
                    feature_name_1 = new_data.columns[i]
                    value_1 = new_data.iloc[0, i]
                    feature_name_2 = new_data.columns[i+1] if i+1 < len(new_data.columns) else ''
                    value_2 = new_data.iloc[0, i+1] if i+1 < len(new_data.columns) else ''

                    # Store the feature name and corresponding value
                    row = [f'{feature_name_1}: {value_1}', f'{feature_name_2}: {value_2}']
                    rows.append(row)

                # Create a DataFrame with feature names and values
                rows_df = pd.DataFrame(rows, columns=['Line 1', 'Line 2'])
                st.dataframe(rows_df, use_container_width=True)

            with col2:
                st.image("Graduate_student.jpg", caption="Graduate Student")  # Display image

        else:
            st.warning(f"Prediction result exceeds expected range: {prediction[0]}")

        # Impact factors ranking display
        st.markdown("<h2 style='color: darkblue;font-size: 24px;'>Impact factors ranking:</h2>", unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns([4, 4, 2, 2, 2])  

        with col1:
            st.markdown("<p style='font-size: 20px;'>1. 2nd semester approved course.</p>", unsafe_allow_html=True)
        with col2:  
            st.markdown("<p style='font-size: 20px;'>2. 1st semester approved course.</p>", unsafe_allow_html=True)
        with col3:  
            st.markdown("<p style='font-size: 20px;'>3. Tuition fee.</p>", unsafe_allow_html=True)
        with col4:  
            st.markdown("<p style='font-size: 20px;'>4. Course type.</p>", unsafe_allow_html=True)
        with col5:  
            st.markdown("<p style='font-size: 20px;'>5. Debtor.</p>", unsafe_allow_html=True)

    except ValueError as e:
        st.error(f"Invalid input value: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

