import tkinter as tk
from tkinter import messagebox
import pickle
from sklearn.preprocessing import StandardScaler
from pycaret.classification import load_model as load_classification_model
import itertools
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# 创建GUI应用程序
class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Prediction App")

        # 输入字段
        self.feature_entries = {}
        self.features_label = tk.Label(root, text="请输入特征值（用逗号分隔）：")
        self.features_label.grid(row=0, column=0, columnspan=2)

        # 动态创建输入框
        self.feature_labels = list(model.feature_names_in_)  # 使用模型的特征名称
        for i, feature in enumerate(self.feature_labels):
            tk.Label(root, text=feature).grid(row=i+1, column=0)
            entry = tk.Entry(root)
            entry.grid(row=i+1, column=1)
            self.feature_entries[feature] = entry

        # 预测按钮
        self.predict_button = tk.Button(root, text="预测", command=self.predict)
        self.predict_button.grid(row=len(self.feature_labels)+1, column=0, columnspan=2)

        # 预测结果显示
        self.result_label = tk.Label(root, text="预测结果：")
        self.result_label.grid(row=len(self.feature_labels)+2, column=0, columnspan=2)

    def predict(self):
        try:
            # 获取输入的特征值
            feature_values = [float(self.feature_entries[feature].get()) for feature in self.feature_labels]

            # 特征值标准化
            scaler = StandardScaler()  # 假设需要重新标准化输入数据
            feature_values_scaled = scaler.fit_transform([feature_values])

            # 预测
            prediction = model.predict(feature_values_scaled)

            # 显示预测结果
            self.result_label.config(text=f"预测结果：{prediction[0]}")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()