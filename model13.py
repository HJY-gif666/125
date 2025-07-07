 import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from PIL import Image  # 添加Image模块

# Page configuration
st.set_page_config(
    page_title='Evaluating and forecasting undergraduate dropouts ',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Title of the app centered with HTML and CSS
st.markdown("""
    <style>
    .center-content {
        text-align: center;
    }
    </style>
    <h1 style='color: darkblue;' class='center-content'>Evaluating and forecasting undergraduate dropouts</h1>
    """, unsafe_allow_html=True)

# Additional info centered
st.markdown("""
    <div class='center-content'>
    Developed by Dr. Songbo Wang et al., Hubei University of Technology .<br>
    Email: Wangsongbo@hbut.edu.cn; 
    </div>
    """, unsafe_allow_html=True)

st.write(' ')
# Open and resize the image
image = Image.open("Figure 4-generic scheme of lap-shear.jpg")
resized_image = image.resize((430, 200))