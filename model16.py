import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from PIL import Image  # 添加Image模块

# 页面配置
st.set_page_config(
    page_title='Evaluating and forecasting undergraduate dropouts ',
    layout='wide',
    initial_sidebar_state='expanded'
)

# 页面标题，居中显示
st.markdown("""
    <style>
    .center-content {
        text-align: center;
    }
    </style>
    <h1 style='color: darkblue;' class='center-content'>Evaluating and forecasting undergraduate dropouts</h1>
    """, unsafe_allow_html=True)

# 页面额外信息，居中显示
st.markdown("""
    <div class='center-content'>
    Developed by Dr. Songbo Wang et al., Hubei University of Technology .<br>
    Email: Wangsongbo@hbut.edu.cn; 
    </div>
    """, unsafe_allow_html=True)

st.write(' ')

# 打开并调整图片大小
try:
    image = Image.open("Domestic and International.jpg")  # 确保图片路径正确
    resized_image = image.resize((567, 449))  # 调整图片大小
    st.image(resized_image, caption="Domestic and international students", use_container_width=True)  # 显示图片
except Exception as e:
    st.error(f"加载图片时出错: {e}")

# Display the image centered
st.image(resized_image, use_column_width=False)

# Input widgets
st.sidebar.subheader('Input features for single prediction')
Marital = st.sidebar.slider('Marital status', 1, 4, 1.08)
Mode = st.sidebar.slider('Application order', 1, 18, 6.74)
Order = st.sidebar.slider("Application order", 1, 5, 1.49)
Course = st.sidebar.slider('Course type', 1, 17, 10.12)
Attendance = st.sidebar.slider('Daytime/evening attendance', 0,1, 0.95)
Qualification = st.sidebar.slider("Previous qualification", 1, 14, 1.74)
Nationality = st.sidebar.slider("Nationality", 1, 21, 1.23)
Mother-Q = st.sidebar.slider("Mother qualification", 1, 28, 11.41)
Father-Q = st.sidebar.slider('Father qualification', 1, 28, 11.78)
Mother-O= st.sidebar.slider('Mother occupation', 1, 25, 7.5)
Father-O = st.sidebar.slider("Father occupation", 1, 26, 8.01)
Displaced = st.sidebar.slider('Displaced', 0, 1, 0.55)
Need = st.sidebar.slider('Educational special need', 0, 1, 0.12)
Debtor = st.sidebar.slider("Debtor", 0, 1, 0.26)
Fee = st.sidebar.slider("Tuition fee", 0, 1, 0.74)
Gender = st.sidebar.slider("Gender", 0, 1, 0.24)
Scholarship = st.sidebar.slider('Scholarship', 0, 1, 0.21)
Age = st.sidebar.slider("Age", 18, 59, 23.09)
First = st.sidebar.slider("1st semester approved course", 0, 18, 4.91)
Second = st.sidebar.slider("2nd semester approved course", 0, 12, 4.31)
Unemployment = st.sidebar.slider("Unemployment rate", 7.6, 16.2, 11.49)
Inflation = st.sidebar.slider("Inflation rate", -0.8,3.7，1.18)
GDP = st.sidebar.slider("GDP", -4.06,3.51,0.34)



# 加载预训练模型
try:
    model1 = pickle.load(open('best_model_non1.pkl', 'rb'))  # 确保模型文件路径正确
    model2 = pickle.load(open('best_model_int.pkl', 'rb'))
except Exception as e:
    st.error(f"加载模型时出错: {e}")
