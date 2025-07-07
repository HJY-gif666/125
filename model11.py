import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 显示封面图片
image_path = '/mnt/data/图片1.tif'
st.image(image_path, caption="欢迎使用模型预测", use_column_width=True)

# 加载预训练模型
try:
    model1 = pickle.load(open('best_model_non1.pkl', 'rb'))  # 确保模型文件路径正确
    model2 = pickle.load(open('best_model_int.pkl', 'rb'))
except Exception as e:
    st.error(f"加载模型时出错: {e}")