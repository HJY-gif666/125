import streamlit as st
import pickle
import pandas as pd
from PIL import Image
from catboost import Pool  # 导入CatBoost的Pool模块
from sklearn.preprocessing import StandardScaler

# 加载预训练模型
model1 = pickle.load(open('best_model_non5.pkl', 'rb'))  # 确保模型文件路径正确
model2 = pickle.load(open('best_model_int.pkl', 'rb'))

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

# Button to trigger single predictions
if st.sidebar.button('Predict'):
    st.markdown("<h1 style='color: blue; font-size: 24px;'>Predicted result:</h1>", unsafe_allow_html=True)

    # 输入特征
    input_features = [[Marital, Mode, Order, Course, Attendance, Qualification, Nationality,
                       Mother_Q, Father_Q, Mother_O, Father_O, Displaced, Need, Debtor,
                       Fee, Gender, Scholarship, Age, First, Second, Unemployment, Inflation, GDP]]

    # 创建 DataFrame
    new_data = pd.DataFrame(input_features, columns=['Marital', 'Mode', 'Order', 'Course', 'Attendance', 'Qualification',
                                                     'Nationality', 'Mother_Q', 'Father_Q', 'Mother_O', 'Father_O',
                                                     'Displaced', 'Need', 'Debtor', 'Fee', 'Gender', 'Scholarship',
                                                     'Age', 'First', 'Second', 'Unemployment', 'Inflation', 'GDP'])

    # 处理缺失值：填充缺失值为0
    new_data = new_data.fillna(0)  # 你也可以选择其他填充方法，如填充均值或中位数

    # 确保数据类型正确
    for column in new_data.select_dtypes(include=['float64']).columns:
        new_data[column] = new_data[column].astype(float)
    for column in new_data.select_dtypes(include=['int64']).columns:
        new_data[column] = new_data[column].astype(int)

    # 加载保存的Scaler对象
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # 对输入数据进行标准化
    feature_values_scaled = scaler.transform(new_data)  # 使用相同的Scaler进行转换

    # 根据Nationality选择模型
    if Nationality == 1:
        # 使用model1预测
        prediction_graduate = model1.predict(feature_values_scaled)  # 预测是否毕业
    else:
        # 使用model2预测
        prediction_graduate = model2.predict(new_data)  # 预测是否毕业

    # 将预测结果添加到DataFrame中
    new_data['Graduate'] = prediction_graduate
    new_data['International'] = 0  # (Add the appropriate logic if needed for this variable)

    # 显示预测结果
    st.write(new_data)
