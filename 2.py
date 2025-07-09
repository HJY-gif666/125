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
    <h1 style='color: darkblue; text-align: center;'>Evaluating and forecasting undergraduate dropouts</h1>
    <div style='text-align: center;'>Developed by Dr. Songbo Wang et al., Hubei University of Technology.</div>
    <div style='text-align: center;'>Email: Wangsongbo@hbut.edu.cn</div>
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
    st.markdown("<h1 style='color: blue; font-size: 24px;'>Predicted result:</h1>", unsafe_allow_html=True)

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

            if prediction[0] == 1:
                st.success("Predicted outcome: Dropout")
            elif prediction[0] == 0:
                st.success("Predicted outcome: Graduated")
                
                # Impact factors ranking display
                st.markdown("<h2 style='font-size: 24px;'>Impact factors ranking:</h2>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 20px;'>1. 2nd semester approved course</p>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 20px;'>2. 1st semester approved course</p>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 20px;'>3. Tuition fee</p>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 20px;'>4. Course type</p>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 20px;'>5. Debtor</p>", unsafe_allow_html=True)

                # Use st.columns() for centering
                col3, col4, col5 = st.columns([4, 1, 3])  # Center column is larger
                with col3:
                    # Modify to display four columns: Feature Name 1, Feature Value 1, Feature Name 2, Feature Value 2
                    rows = []
                    for i in range(0, len(new_data.columns), 2):
                        feature_name_1 = new_data.columns[i]
                        value_1 = new_data.iloc[0, i]
                        feature_name_2 = new_data.columns[i+1] if i+1 < len(new_data.columns) else ''
                        value_2 = new_data.iloc[0, i+1] if i+1 < len(new_data.columns) else ''

                        # Modify the display of feature names and values to use 20px font size
                        row = [
                            f"<span style='font-size: 20px;'>{feature_name_1}</span>", 
                            f"<span style='font-size: 20px;'>{value_1}</span>", 
                            f"<span style='font-size: 20px;'>{feature_name_2}</span>" if feature_name_2 else "", 
                            f"<span style='font-size: 20px;'>{value_2}</span>" if value_2 else ""
                        ]
                        rows.append(row)

                    # Create DataFrame with four columns: Feature Name 1, Feature Value 1, Feature Name 2, Feature Value 2
                    rows_df = pd.DataFrame(rows, columns=['Feature Name 1', 'Feature Value 1', 'Feature Name 2', 'Feature Value 2'])

                    # Display the DataFrame with 20px font size for the text in Streamlit
                    st.dataframe(rows_df.style.set_properties(**{'font-size': '20px'}), use_container_width=True)

                with col5:
                    st.image("Graduate_student.jpg", caption="")  # Display image in the center column

            else:
                st.warning(f"Prediction result exceeds expected range: {prediction[0]}")
        else:  # Otherwise use model1
            prediction = model1.predict(feature_values_scaled)
 
            if prediction[0] == 1:
                st.success("Predicted outcome: Dropout")
            elif prediction[0] == 0:
                st.success("Predicted outcome: Graduated")
                
                # Impact factors ranking display
                st.markdown("<h2 style='font-size: 24px;'>Impact factors ranking:</h2>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 20px;'>1. 2nd semester approved course</p>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 20px;'>2. 1st semester approved course</p>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 20px;'>3. Tuition fee</p>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 20px;'>4. Course type</p>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 20px;'>5. Debtor</p>", unsafe_allow_html=True)

                # Use st.columns() for centering
                col3, col4, col5 = st.columns([3, 1, 3])  # Center column is larger
                with col3:
                    # Modify to display four columns: Feature Name 1, Feature Value 1, Feature Name 2, Feature Value 2
                    rows = []
                    for i in range(0, len(new_data.columns), 2):
                        feature_name_1 = new_data.columns[i]
                        value_1 = new_data.iloc[0, i]
                        feature_name_2 = new_data.columns[i+1] if i+1 < len(new_data.columns) else ''
                        value_2 = new_data.iloc[0, i+1] if i+1 < len(new_data.columns) else ''

                        # Modify the display of feature names and values to use 20px font size
                        row = [
                            f"<span style='font-size: 20px;'>{feature_name_1}</span>", 
                            f"<span style='font-size: 20px;'>{value_1}</span>", 
                            f"<span style='font-size: 20px;'>{feature_name_2}</span>" if feature_name_2 else "", 
                            f"<span style='font-size: 20px;'>{value_2}</span>" if value_2 else ""
                        ]
                        rows.append(row)

                    # Create DataFrame with four columns: Feature Name 1, Feature Value 1, Feature Name 2, Feature Value 2
                    rows_df = pd.DataFrame(rows, columns=['Feature Name 1', 'Feature Value 1', 'Feature Name 2', 'Feature Value 2'])

                    # Display the DataFrame with 20px font size for the text in Streamlit
                    st.dataframe(rows_df.style.set_properties(**{'font-size': '20px'}), use_container_width=True)

                with col5:
                    st.image("Graduate_student.jpg", caption="")  # Display image in the center column

            else:
                st.warning(f"Prediction result exceeds expected range: {prediction[0]}")

    except ValueError as e:
        st.error(f"Invalid input value: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")


