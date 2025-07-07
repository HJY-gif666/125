import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 加载预训练模型
try:
    model1 = pickle.load(open('best_model_non1.pkl', 'rb'))  # 确保模型文件路径正确
    model2 = pickle.load(open('best_model_int.pkl', 'rb'))
 

# 加载CSV文件，获取特征名称和范围
df = pd.read_csv('Dataset-Non.csv')  # 加载上传的CSV文件

# 获取特征列（假设最后一列是目标列）
input_columns = df.columns[:-1]  # 获取所有特征（不包括目标列）
target_column = df.columns[-1]   # 假设最后一列是目标列

# 获取每个特征的最小值和最大值
feature_ranges = df[input_columns].describe().transpose()[['min', 'max']]  # 获取每个特征的最小值和最大值

# 创建Streamlit界面
st.title("模型预测应用")

# 使用列布局展示标题和图片
col1, col2 = st.columns([2, 1])

with col1:
    st.image("Domestic and International.tif", caption="1")  # 更新图片路径
with col2:

    st.markdown("由：湖北工业大学 王松波博士")
    st.markdown("邮箱：Wangsonbo@hbut.edu.cn | 2110600205@hbut.edu.cn")

# 显示特征范围的表格
st.write("### 各输入特征的有效范围如下：")
st.dataframe(feature_ranges)

# 创建输入框，确保每个特征的值在有效范围内，使用滑块
feature_values = []
label_encoder = LabelEncoder()

for feature in input_columns:
    min_val = feature_ranges.loc[feature, 'min']
    max_val = feature_ranges.loc[feature, 'max']

    # 对于类别特征，使用LabelEncoder转换为整数
    if df[feature].dtype == 'object':  # 类别数据
        st.write(f"{feature} 是一个类别特征，将被编码为整数。")
        df[feature] = label_encoder.fit_transform(df[feature])  # 转换为整数
        user_input = st.selectbox(f"为 {feature} 选择一个值", options=df[feature].unique())
    
    else:  # 对于数值特征
        if feature == 'Nationality':  # 对于国籍，限制输入在1到21之间
            user_input = st.slider(f"为 {feature} 选择一个值 (最小值: 1, 最大值: 21)", 
                                   min_value=1, max_value=21, value=1, step=1)
            st.write(f"当前输入值为 {feature} 是: {user_input}")
        
        elif feature in ['Unemployment', 'Inflation']:  # 保留一位小数
            user_input = st.slider(f"为 {feature} 选择一个值 (最小值: {min_val}, 最大值: {max_val})", 
                                   min_value=float(min_val), max_value=float(max_val), value=float(min_val), step=0.1)
            st.write(f"当前输入值为 {feature} 是: {round(user_input, 1):.1f}")
        
        elif feature == 'GDP':  # 保留两位小数
            user_input = st.slider(f"为 {feature} 选择一个值 (最小值: {min_val}, 最大值: {max_val})", 
                                   min_value=float(min_val), max_value=float(max_val), value=float(min_val), step=0.01)
            st.write(f"当前输入值为 {feature} 是: {round(user_input, 2):.2f}")
        
        else:  # 其他数值特征，整数输入
            user_input = st.slider(f"为 {feature} 选择一个值 (最小值: {min_val}, 最大值: {max_val})", 
                                   min_value=int(min_val), max_value=int(max_val), value=int(min_val), step=1)
        
    feature_values.append(user_input)

if st.button("预测"):
    try:
        # 加载保存的Scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)  # 加载Scaler对象

        # 使用Scaler进行标准化
        feature_values_scaled = scaler.transform([feature_values])  # 使用相同的Scaler进行转换

        # 检查Nationality特征是否在有效范围内
        nationality_index = df.columns.get_loc('Nationality')  # 获取Nationality列的索引
        nationality_value = feature_values[nationality_index]  # 获取输入的Nationality值

        if 2 <= nationality_value <= 21:  # 如果Nationality值在2到21之间
            # 使用best_model_int.pkl进行预测
            prediction = model2.predict(feature_values_scaled)  # 使用整数模型进行预测
            st.write("使用best_model_int.pkl的预测结果:", prediction)

            if prediction[0] == 0:
                st.success(f"预测的目标值：未毕业")
            elif prediction[0] == 1:
                st.success(f"预测的目标值：已毕业")
                # 如果预测结果为1（已毕业），显示第二张图片（Graduate.png）
                st.image("https://github.com/yourusername/yourrepo/raw/main/images/Graduate.png", caption="已毕业！")
            else:
                st.warning(f"预测结果超出预期范围: {prediction[0]}")
        else:
            # 使用best_model_non1.pkl进行预测
            prediction = model1.predict(feature_values_scaled)  # 使用非整数模型进行预测
            st.write("使用best_model_non1.pkl的预测结果:", prediction)

            if prediction[0] == 0:
                st.success(f"预测的目标值：未毕业")
            elif prediction[0] == 1:
                st.success(f"预测的目标值：已毕业")
                # 如果预测结果为1（已毕业），显示第二张图片（Graduate.png）
                st.image("https://github.com/yourusername/yourrepo/raw/main/images/Graduate.png", caption="已毕业！")
            else:
                st.warning(f"预测结果超出预期范围: {prediction[0]}")

    except ValueError as e:
        st.error(f"无效的输入值: {e}")
    except Exception as e:
        st.error(f"发生错误: {e}")