import streamlit as st
import pickle
import pandas as pd
from catboost import Pool  # 导入CatBoost的Pool模块

# 加载预训练模型
model1 = pickle.load(open('best_model_non1.pkl', 'rb'))  # 确保模型文件路径正确
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

# 输入特征
st.sidebar.subheader('Input features for single prediction')
Marital = st.sidebar.slider('Marital status', 1, 4, 1)
Mode = st.sidebar.slider('Application order', 1, 18, 9)
Order = st.sidebar.slider("Application order", 1, 5, 2)
Course = st.sidebar.slider('Course type', 1, 17, 10)
Attendance = st.sidebar.slider('Daytime/evening attendance', 0, 1, 0)
Qualification = st.sidebar.slider("Previous qualification", 1, 14, 7)
Nationality = st.sidebar.slider("Nationality", 1, 21, 10)
Mother-Q = st.sidebar.slider("Mother qualification", 1, 28, 14)
Father-Q = st.sidebar.slider('Father qualification', 1, 28, 15)
Mother-O = st.sidebar.slider('Mother occupation', 1, 25, 18)
Father-O = st.sidebar.slider("Father occupation", 1, 26, 20)
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
Inflation = st.sidebar.slider("Inflation rate", -0.8, 3.7, 1)
GDP = st.sidebar.slider("GDP", -4.06, 3.51, 0)

# Button to trigger single predictions
if st.sidebar.button('Predict'):
    st.markdown("<h1 style='color: blue; font-size: 24px;'>Predicted result:</h1>", unsafe_allow_html=True)
    
    # 输入特征
    input_features = [[Marital, Mode, Order, Course, Attendance, Qualification, Nationality,
                       Mother-Q, Father-Q, Mother-O, Father-O, Displaced, Need, Debtor,
                       Fee, Gender, Scholarship, Age, First, Second, Unemployment, Inflation, GDP]]

    # 创建 DataFrame
    new_data = pd.DataFrame(input_features, columns=['Marital', 'Mode', 'Order', 'Course', 'Attendance', 'Qualification', 
                                                     'Nationality', 'Mother-Q', 'Father-Q', 'Mother-O', 'Father-O', 
                                                     'Displaced', 'Need', 'Debtor', 'Fee', 'Gender', 'Scholarship', 
                                                     'Age', 'First', 'Second', 'Unemployment', 'Inflation', 'GDP'])

    # 将数据转换为CatBoost的Pool格式
    pool = Pool(new_data)

    # 使用CatBoost模型进行预测
    prediction_graduate = model1.predict(pool)  # 预测是否毕业
    prediction_international = model2.predict(pool)  # 预测是否国际学生

    # 将预测结果添加到DataFrame中
    new_data['Graduate'] = prediction_graduate
    new_data['International'] = prediction_international

    # 显示预测结果
    st.write(new_data)
