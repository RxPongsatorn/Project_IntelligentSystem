import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# ตั้งค่าหน้าเว็บ
st.title("Machine Learning with Streamlit")

# โหลดข้อมูล
path = "data/LoanDataset - LoansDatasest.csv"
df = pd.read_csv(path)

# จัดการค่าของข้อมูล
df['loan_amnt'] = df['loan_amnt'].replace({'£': '', ',': ''}, regex=True).astype(float)
df['customer_income'] = df['customer_income'].str.replace(",", "").astype(int)
df = df.dropna()

st.write("### Data Preview:")
st.write(df.head())  # แสดงข้อมูลตัวอย่าง

# เลือก Feature ที่เหมาะสมกับ X และ Y
x_feature = "customer_income"
y_feature = "loan_amnt"

X = df[[x_feature]]
y = df[y_feature]

# Sidebar - เลือกค่าพารามิเตอร์
st.sidebar.header("Model Parameters")

# เลือกขนาด Train/Test Split
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, step=5) / 100

# เลือกค่า k สำหรับ KNN
k = st.sidebar.slider("เลือกค่า K สำหรับ KNN", 1, 15, 3)

# เลือกจำนวน Clusters สำหรับ K-Means
num_clusters = st.sidebar.slider("เลือกจำนวน Clusters", 2, 10, 3)

# Standardize ข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# แบ่งข้อมูล Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# KNN Classification
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_scaled)  # ใช้ข้อมูลทั้งหมด

# K-Means Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Plot KNN Classification
fig, ax = plt.subplots()
sns.scatterplot(x=df[x_feature], y=df[y_feature], hue=knn_pred, palette='viridis', ax=ax)
ax.set_title(f"KNN Classification (k={k})")
ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
st.pyplot(fig)

# Plot K-Means Clustering
fig, ax = plt.subplots()
sns.scatterplot(x=df[x_feature], y=df[y_feature], hue=cluster_labels, palette='tab10', ax=ax)
ax.set_title(f"K-Means Clustering (Clusters={num_clusters})")
ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
st.pyplot(fig)
