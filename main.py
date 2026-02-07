import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from lazypredict.Supervised import LazyClassifier
from sklearn.linear_model import LogisticRegression

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import os

class SleepDisorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sleep Disorder Classification")
        self.root.geometry("800x600")

        self.filename = "C:/Users/user/PycharmProjects/sleepdisorder/Sleep_health_and_lifestyle_dataset.csv"
        self.data = pd.read_csv(self.filename)

        if os.path.exists("background.jpg"):
            self.bg_image = Image.open("background.jpg")
            self.bg_photo = ImageTk.PhotoImage(self.bg_image.resize((800, 600)))
            self.background_label = tk.Label(self.root, image=self.bg_photo)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        else:
            self.root.configure(bg='lightblue')

        self.create_widgets()
        threading.Thread(target=self.preprocess_and_train).start()

    def create_widgets(self):
        frame = tk.Frame(self.root, bg='white', bd=2)
        frame.place(relx=0.5, rely=0.5, anchor='center')

        ttk.Label(frame, text="Sleep Disorder Detection", font=("Helvetica", 18)).pack(pady=15)

        self.status_label = ttk.Label(frame, text=f"Loaded: {self.filename}", foreground="green")
        self.status_label.pack()

        ttk.Button(frame, text="📁 Dataset", command=self.show_data).pack(pady=5)
        ttk.Button(frame, text="📊 Countplot", command=self.show_countplot).pack(pady=5)
        ttk.Button(frame, text="📊 Correlation Heatmap", command=self.show_heatmap).pack(pady=5)
        ttk.Button(frame, text="📊 LazyClassifier Accuracy", command=self.show_lazy_accuracy).pack(pady=5)
        ttk.Button(frame, text="📊 Confusion Matrix", command=self.show_confusion_matrix).pack(pady=5)
        ttk.Button(frame, text="📊 Lazy Accuracy After Cleaning", command=self.show_lazy_accuracy_after_cleaning).pack(pady=5)
        ttk.Button(frame, text="📊 Confusion Matrix After Cleaning", command=self.show_confusion_matrix_after_cleaning).pack(pady=5)

    def preprocess_and_train(self):
        try:
            self.data.dropna(inplace=True)
            label_encoder = LabelEncoder()
            for col in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
                self.data[col] = label_encoder.fit_transform(self.data[col])
            self.data[['Systolic BP', 'Diastolic BP']] = self.data['Blood Pressure'].str.split('/', expand=True)
            self.data[['Systolic BP', 'Diastolic BP']] = self.data[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric)
            self.data.drop(['Blood Pressure', 'Person ID'], axis=1, inplace=True)

            self.X = self.data.drop('Sleep Disorder', axis=1)
            self.y = self.data['Sleep Disorder']

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)

            self.clf = LazyClassifier()
            self.models, _ = self.clf.fit(self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test)

            self.log_reg = LogisticRegression(max_iter=1000)
            self.log_reg.fit(self.X_train_scaled, self.y_train)
            self.y_pred = self.log_reg.predict(self.X_test_scaled)

            self.cleaned_data = self.data.copy()
            num_col = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
                       'Heart Rate', 'Daily Steps', 'Systolic BP', 'Diastolic BP']
            Q1 = self.cleaned_data[num_col].quantile(0.25)
            Q3 = self.cleaned_data[num_col].quantile(0.75)
            IQR = Q3 - Q1
            self.cleaned_data = self.cleaned_data[~((self.cleaned_data[num_col] < (Q1 - 1.5 * IQR)) | (self.cleaned_data[num_col] > (Q3 + 1.5 * IQR))).any(axis=1)]

            self.X_clean = self.cleaned_data.drop('Sleep Disorder', axis=1)
            self.y_clean = self.cleaned_data['Sleep Disorder']
            self.X_train_clean, self.X_test_clean, self.y_train_clean, self.y_test_clean = train_test_split(self.X_clean, self.y_clean, test_size=0.2, random_state=42)
            self.X_train_clean_scaled = self.scaler.fit_transform(self.X_train_clean)
            self.X_test_clean_scaled = self.scaler.transform(self.X_test_clean)
            self.models_clean, _ = self.clf.fit(self.X_train_clean_scaled, self.X_test_clean_scaled, self.y_train_clean, self.y_test_clean)

            self.log_reg_clean = LogisticRegression(max_iter=1000)
            self.log_reg_clean.fit(self.X_train_clean_scaled, self.y_train_clean)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_data(self):
        try:
            top = tk.Toplevel(self.root)
            top.title("Dataset Preview")
            text = tk.Text(top, wrap='none')
            text.insert('1.0', self.data.head(20).to_string())
            text.pack(expand=True, fill='both')
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_countplot(self):
        try:
            plt.figure(figsize=(8, 4))
            sns.countplot(x='Sleep Disorder', data=self.data)
            plt.title('Distribution of Sleep Disorders')
            plt.xlabel('Sleep Disorder Type')
            plt.ylabel('Count')
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_heatmap(self):
        try:
            plt.figure(figsize=(12, 8))
            sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('Correlation Heatmap')
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_lazy_accuracy(self):
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(y=self.models.index, x=self.models['Accuracy'], palette='viridis')
            plt.title('Model Accuracy Comparison (LazyClassifier)')
            plt.xlabel('Accuracy')
            plt.ylabel('Models')
            plt.xlim(0, 1)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_confusion_matrix(self):
        try:
            ConfusionMatrixDisplay.from_estimator(self.log_reg, self.X_test_scaled, self.y_test, cmap='Blues')
            plt.title('Confusion Matrix - Logistic Regression')
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_lazy_accuracy_after_cleaning(self):
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(y=self.models_clean.index, x=self.models_clean['Accuracy'], palette='rocket')
            plt.title('Model Accuracy After Outlier Removal (LazyClassifier)')
            plt.xlabel('Accuracy')
            plt.ylabel('Models')
            plt.xlim(0, 1)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_confusion_matrix_after_cleaning(self):
        try:
            ConfusionMatrixDisplay.from_estimator(self.log_reg_clean, self.X_test_clean_scaled, self.y_test_clean, cmap='Greens')
            plt.title('Confusion Matrix - Logistic Regression (After Outlier Removal)')
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = SleepDisorderApp(root)
    root.mainloop()