import streamlit as st
import pickle
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler

# 加载模型
model1 = pickle.load(open('best_model_non5.pkl', 'rb'))
model2 = pickle.load(open('best_model_int7.9.pkl', 'rb'))

# 加载Scaler对象（只加载一次）
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 页面配置
st.set_page_config(
    page_title='Evaluating and forecasting undergraduate dropouts',
    layout='wide',
    initial_sidebar_state='expanded'
)

# 页面标题和额外信息
st.markdown("""
    <h1 style='color: darkblue; text-align: center;'>Evaluating and forecasting undergraduate dropouts</h1>
    <div style='text-align: center;'>Developed by Dr. Songbo Wang et al., Hubei University of Technology.</div>
    <div style='text-align: center;'>Email: Wangsongbo@hbut.edu.cn</div>
""", unsafe_allow_html=True)

# Open and resize the image
image = Image.open("Domestic and International.jpg")
resized_image = image.resize((567, 449))
# Display the image centered
st.markdown("""
    <div style='text-align: center;'>
        <img src="Domestic and International.jpg" width="567" height="449">
    </div>
""", unsafe_allow_html=True)

# 输入特征
st.sidebar.subheader('Input features for single prediction')
Marital = st.sidebar.slider('Marital status', 1, 4, 1)
Mode = st.sidebar.slider('Application order', 1, 18, 9)
Order = st.sidebar.slider("Application order", 1, 5, 2)
Course = st.sidebar.slider('Course type', 1, 17, 10)
Attendance = st.sidebar.slider('Daytime/evening attendance', 0, 1, 0)
Qualification = st.sidebar.slider("Previous qualification", 1, 14, 7)
Nationality = st.sidebar.slider("Nationality", 1, 21, 10)
Mother_Q = st.sidebar.slider("Mother qualification", 1, 28, 14)
Father_Q = st.sidebar.slider('Father qualification', 1, 28, 15)
Mother_O = st.sidebar.slider('Mother occupation', 1, 25, 18)
Father_O = st.sidebar.slider("Father occupation", 1, 26, 20)
Displaced = st.sidebar.slider('Displaced', 0, 1, 0)
Need = st.sidebar.slider('Educational special need', 0, 1, 0)
Debtor = st.sidebar.slider("Debtor", 0, 1, 0)
Fee = st.sidebar.slider("Tuition fee", 0, 1, 0)
Gender = st.sidebar.slider("Gender", 0, 1, 0)
Scholarship = st.sidebar.slider('Scholarship', 0, 1, 0)
Age = st.sidebar.slider("Age", 18, 59, 23)
First = st.sidebar.slider("1st semester approved course", 0, 18, 9)
Second = st.sidebar.slider("2nd semester approved course", 0, 12, 4)
Unemployment = st.sidebar.slider("Unemployment rate", 7.6, 16.2, 11.0)
Inflation = st.sidebar.slider("Inflation rate", -0.8, 3.7, 1.0)
GDP = st.sidebar.slider("GDP", -4.06, 3.51, 1.00)

# 按钮触发预测
if st.sidebar.button('Predict'):
    st.markdown("<h1 style='color: blue; font-size: 24px;'>Predicted result:</h1>", unsafe_allow_html=True)

    # 输入特征进行预测
    input_features = [[Marital, Mode, Order, Course, Attendance, Qualification, Nationality,
                       Mother_Q, Father_Q, Mother_O, Father_O, Displaced, Need, Debtor,
                       Fee, Gender, Scholarship, Age, First, Second, Unemployment, Inflation, GDP]]
    
    # 创建 DataFrame
    new_data = pd.DataFrame(input_features, columns=['Marital', 'Mode', 'Order', 'Course', 'Attendance', 'Qualification',
                                                     'Nationality', 'Mother_Q', 'Father_Q', 'Mother_O', 'Father_O',
                                                     'Displaced', 'Need', 'Debtor', 'Fee', 'Gender', 'Scholarship',
                                                     'Age', 'First', 'Second', 'Unemployment', 'Inflation', 'GDP'])
# 对输入特征进行标准化
    feature_values_scaled = scaler.transform(new_data.values)

    print(feature_values_scaled.shape)    

# 打印 new_data 的列名和数据，检查是否有 'Nationality' 列
st.write("Data Columns:", new_data.columns)  # 输出列名
st.write("Data Preview:", new_data.head())  # 输出 DataFrame 的前几行

# 检查是否包含 'Nationality' 列
if 'Nationality' in new_data.columns:
    nationality_value = new_data.iloc[0]['Nationality']  # 使用 iloc 安全访问
    st.write(f"Nationality: {nationality_value}")
else:
    st.write("The 'Nationality' column does not exist in the data.")

# 使用模型进行预测
if nationality_value == 1:
    prediction1 = model1.predict(feature_values_scaled)
    st.write(f"使用 Model 1 预测结果: {prediction1[0]}")
else:
    prediction2 = model2.predict(feature_values_scaled)
    st.write(f"使用 Model 2 预测结果: {prediction2[0]}")

