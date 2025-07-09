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
# 使用 columns 来分成两列
col1, col2 = st.sidebar.columns(2)

with col1:
    Marital = st.slider('Marital status', 1, 4, 1)
    Mode = st.slider('Application order', 1, 18, 9)
    Order = st.slider("Application order", 1, 5, 2)
    Course = st.slider('Course type', 1, 17, 10)
    Attendance = st.slider('Daytime/evening attendance', 0, 1, 0)
    Qualification = st.slider("Previous qualification", 1, 14, 7)
    Nationality = st.slider("Nationality", 1, 21, 1)
    Mother_Q = st.slider("Mother qualification", 1, 28, 14)
    Father_Q = st.slider('Father qualification', 1, 28, 15)
    Mother_O = st.slider('Mother occupation', 1, 25, 18)
    Father_O = st.slider("Father occupation", 1, 26, 20)
    Displaced = st.slider('Displaced', 0, 1, 0)

with col2:
    Need = st.slider('Educational special need', 0, 1, 0)
    Debtor = st.slider("Debtor", 0, 1, 0)
    Fee = st.slider("Tuition fee", 0, 1, 0)
    Gender = st.slider("Gender", 0, 1, 0)
    Scholarship = st.slider('Scholarship', 0, 1, 0)
    Age = st.slider("Age", 18, 59, 23)
    First = st.slider("1st semester approved course", 0, 18, 9)
    Second = st.slider("2nd semester approved course", 0, 12, 4)
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

    # 修改为每行两个特征值
    rows = []
    for i in range(0, len(new_data.columns), 2):
        row = new_data.iloc[0, i:i+2].values  # Get two features per row
        rows.append(row)
    
    # Create a DataFrame with two columns per row
    rows_df = pd.DataFrame(rows, columns=['Feature 1', 'Feature 2'])

    # Display the table with two columns per row
    st.write("### Input Features")
    st.dataframe(rows_df, use_container_width=True)

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
                col3, col4, col5 = st.columns([3, 4, 2])  # 中间的列宽度为4，左右两列宽度为1
                with col3:
                    st.write("Impact factors ranking:")
                    st.write("1. Second")
                    st.write("2. First")
                    st.write("3. Fee")
                    st.write("4. Course")
                    st.write("5. Debtor")

                with col4:
                    st.image("Graduate_student.jpg", caption="  ")  # 显示图片在中间的列


            else:
                st.warning(f"预测结果超出预期范围: {prediction[0]}")
        else:  # 否则使用 model1
            prediction = model1.predict(feature_values_scaled)
 
            if prediction[0] == 1:
                st.success("Predicted outcome: Dropout")
            elif prediction[0] == 0:
                st.success("Predicted outcome: Graduated")
                
                # 使用st.columns()进行居中
                col3, col4, col5 = st.columns([3, 4, 2])  # 中间的列宽度为4，左右两列宽度为1
                with col3:
                    st.write("Impact factors ranking:")
                    st.write("1. Second")
                    st.write("2. First")
                    st.write("3. Fee")
                    st.write("4. Course")
                    st.write("5. Debtor")
                    
                with col4:
                    st.image("Graduate_student.jpg", caption="  ")  # 显示图片在中间的列

            else:
                st.warning(f"预测结果超出预期范围: {prediction[0]}")

    except ValueError as e:
        st.error(f"无效的输入值: {e}")
    except Exception as e:
        st.error(f"发生错误: {e}")
