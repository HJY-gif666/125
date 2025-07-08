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
    resized_image = image.resize((430, 200))  # 调整图片大小
    st.image(resized_image, caption="Domestic and international students", use_container_width=True)  # 显示图片
except Exception as e:
    st.error(f"加载图片时出错: {e}")

# 加载预训练模型
try:
    model1 = pickle.load(open('best_model_non1.pkl', 'rb'))  # 确保模型文件路径正确
    model2 = pickle.load(open('best_model_int.pkl', 'rb'))
except Exception as e:
    st.error(f"加载模型时出错: {e}")
