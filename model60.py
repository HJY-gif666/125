import streamlit as st
import pickle
import pandas as pd
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

st.sidebar.subheader('Input features for single prediction')
# 使用 columns 来分成六列，且每列宽度一致
col1, col2, col3, col4, col5, col6 = st.sidebar.columns([1, 1, 1, 1, 1, 1])

with col1:
    Marital = st.slider('Marital status', 1, 4, 1)
    Mode = st.slider('Application order', 1, 18, 9)
    Order = st.slider("Application order", 1, 5, 2)
    Course = st.slider('Course type', 1, 17, 10)

with col2:
    Attendance = st.slider('Daytime/evening attendance', 0, 1, 0)
    Qualification = st.slider("Previous qualification", 1, 14, 7)
    Nationality = st.slider("Nationality", 1, 21, 1)
    Mother_Q = st.slider("Mother qualification", 1, 28, 14)

with col3:
    Father_Q = st.slider('Father qualification', 1, 28, 15)
    Mother_O = st.slider('Mother occupation', 1, 25, 18)
    Father_O = st.slider("Father occupation", 1, 26, 20)
    Displaced = st.slider('Displaced', 0, 1, 0)

with col4:
    Need = st.slider('Educational special need', 0, 1, 0)
    Debtor = st.slider("Debtor", 0, 1, 0)
    Fee = st.slider("Tuition fee", 0, 1, 0)
    Gender = st.slider("Gender", 0, 1, 0)

with col5:
    Scholarship = st.slider('Scholarship', 0, 1, 0)
    Age = st.slider("Age", 18, 59, 23)
    First = st.slider("1st semester approved course", 0, 18, 9)
    Second = st.slider("2nd semester approved course", 0, 12, 4)

with col6:
    Unemployment = st.slider("Unemployment rate", 7.6, 16.2, 11.0)
    Inflation = st.slider("Inflation rate", -0.8, 3.7, 1.0)
    GDP = st.slider("GDP", -4.06, 3.51, 1.00)

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

    # 使用 CSS 调整表格列宽
    st.markdown("""
        <style>
            .streamlit-table th, .streamlit-table td {
                width: 150px;  /* 固定列宽 */
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    # 将特征拆分为六部分显示
    split_index = len(new_data.columns) // 6  # 计算每部分的拆分点
    data_part_1 = new_data.iloc[:, :split_index]  # 获取前六分之一列
    data_part_2 = new_data.iloc[:, split_index:2*split_index]  # 获取第二分列
    data_part_3 = new_data.iloc[:, 2*split_index:3*split_index]  # 获取第三分列
    data_part_4 = new_data.iloc[:, 3*split_index:4*split_index]  # 获取第四分列
    data_part_5 = new_data.iloc[:, 4*split_index:5*split_index]  # 获取第五分列
    data_part_6 = new_data.iloc[:, 5*split_index:]  # 获取第六分列

    # 显示输入特征的表格
    st.write("### Input Features")
    st.dataframe(data_part_1, use_container_width=True)  # 显示第一部分的表格
    st.dataframe(data_part_2, use_container_width=True)  # 显示第二部分的表格
    st.dataframe(data_part_3, use_container_width=True)  # 显示第三部分的表格
    st.dataframe(data_part_4, use_container_width=True)  # 显示第四部分的表格
    st.dataframe(data_part_5, use_container_width=True)  # 显示第五部分的表格
    st.dataframe(data_part_6, use_container_width=True)  # 显示第六部分的表格

    # 对输入特征进行标准化
    feature_values_scaled = scaler.transform(new_data.values)

    # 获取 Nationality 特征的值
    nationality_value = new_data.iloc[0]['Nationality']
    st.write(f"Nationality: {nationality_value}")  # 打印 Nationality 值

    try:
        # 判断使用哪个模型进行预测
        if 2 <= nationality_value <= 21:  # 如果 Nationality 值在 2 到 21 之间，使用 model2
            prediction = model2.predict(feature_values_scaled)

            if prediction[0] == 1:
                st.success("Predicted outcome: Dropout")
            elif prediction[0] == 0:
                st.success("Predicted outcome: Graduated")
                
                # 使用st.columns()进行居中
                col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])  # 6列，宽度相等

                with col3:  # 中间列显示图片
                    st.image("Graduate_student.jpg", caption="Graduated!")

            else:
                st.warning(f"预测结果超出预期范围: {prediction[0]}")
        else:  # 否则使用 model1
            prediction = model1.predict(feature_values_scaled)
 
            if prediction[0] == 1:
                st.success("Predicted outcome: Dropout")
            elif prediction[0] == 0:
                st.success("Predicted outcome: Graduated")
                
                # 使用st.columns()进行居中
                col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])  # 6列，宽度相等

                with col3:  # 中间列显示图片
                    st.image("Graduate_student.jpg", caption="Graduated!")

            else:
                st.warning(f"预测结果超出预期范围: {prediction[0]}")

    except ValueError as e:
        st.error(f"无效的输入值: {e}")
    except Exception as e:
        st.error(f"发生错误: {e}")
