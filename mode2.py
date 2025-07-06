import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 加载训练好的模型
model = pickle.load(open('best_model.pkl', 'rb'))  # 请确保模型文件路径正确

# 读取CSV文件来获取特征名称
df = pd.read_csv('Dataset-Non.csv')  # 加载上传的 CSV 文件

# 自动获取输入特征（假设最后一列是目标列）
input_columns = df.columns[:-1]  # 获取除目标列外的所有特征列
target_column = df.columns[-1]   # 假设最后一列是目标列（Target）

# 创建Streamlit界面
st.title("模型预测应用程序")

# 输入特征
st.write("请输入特征值（用逗号分隔）：")

# 动态创建输入框
feature_values = []
for feature in input_columns:
    value = st.number_input(f"请输入 {feature} 的值", value=0.0)
    feature_values.append(value)

# 预测按钮
if st.button("预测"):
    try:
        # 特征值标准化
        scaler = StandardScaler()  # 假设需要重新标准化输入数据
        feature_values_scaled = scaler.fit_transform([feature_values])

        # 进行预测
        prediction = model.predict(feature_values_scaled)

        # 显示预测结果
        st.success(f"预测结果：{prediction[0]}")
    except ValueError:
        st.error("请输入有效的数值")

