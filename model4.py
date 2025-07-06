import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 加载预训练的模型
try:
    model1 = pickle.load(open('best_model_non1.pkl', 'rb'))  # 确保模型文件路径正确
    model2 = pickle.load(open('best_model_int.pkl', 'rb'))
    st.write("模型加载成功！")
except Exception as e:
    st.error(f"模型加载失败: {e}")

# 加载CSV文件，获取特征名称和范围
df = pd.read_csv('Dataset-Non.csv')  # 加载上传的CSV文件

# 获取特征列（假设最后一列是目标列）
input_columns = df.columns[:-1]  # 获取除目标列外的所有特征
target_column = df.columns[-1]   # 假设最后一列是目标列（Target）

# 获取特征的最小值和最大值
feature_ranges = df[input_columns].describe().transpose()[['min', 'max']]  # 获取每个特征的最小值和最大值

# 创建Streamlit界面
st.title("模型预测应用程序")

# 显示特征范围
st.write("每个输入特征的有效范围如下：")
st.dataframe(feature_ranges)

# 创建输入框，确保输入的值在特征的有效范围内
feature_values = []
for feature in input_columns:
    min_val = feature_ranges.loc[feature, 'min']
    max_val = feature_ranges.loc[feature, 'max']
    user_input = st.number_input(f"请输入 {feature} 的值 (最小值: {min_val}, 最大值: {max_val})", 
                                value=float(min_val), 
                                min_value=float(min_val), 
                                max_value=float(max_val))
    feature_values.append(user_input)

# 预测按钮
if st.button("预测"):
    try:
        # 特征标准化（确保使用相同的数据集来拟合标准化器）
        # 假设训练时已经使用了标准化，使用训练时的标准化器进行转换
        scaler = StandardScaler()
        feature_values_scaled = scaler.fit_transform([feature_values])  # 确保输入是二维数组

        # 确保预测输入的形状是二维数组
        feature_values_scaled = np.array(feature_values_scaled).reshape(1, -1)

        # 使用模型进行预测
        prediction = model1.predict(feature_values_scaled)  # 使用适当的模型进行预测

        # 显示预测结果
        st.success(f"预测结果：{prediction[0]}")

    except ValueError as e:
        st.error(f"无效的输入值：{e}")
    except Exception as e:
        st.error(f"发生了错误：{e}")