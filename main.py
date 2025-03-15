import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ตั้งค่าหน้าเว็บ
st.title("Machine Learning with Streamlit")
st.write("Upload dataset แล้วให้ระบบ Train โมเดล และแสดงผล")

path = "data/LoanDataset - LoansDatasest.csv"
# โหลดข้อมูล
df = pd.read_csv(path)
st.write("### Data Preview:")
df['loan_amnt'] = df['loan_amnt'].replace({'£': '', ',': ''}, regex=True).astype(float)
df['customer_income'] = df['customer_income'].str.replace(",", "").astype(int)
df = df.dropna()
print(df.dtypes)

st.write(df.head())  # แสดงข้อมูลตัวอย่าง

# เลือก Feature และ Target
features = st.multiselect("เลือก Features", df.columns, default=df.columns[:-1])
target = st.selectbox("เลือก Target", df.columns)


if features and target:
    X = df[features]
    y = df[target]

    # แบ่งข้อมูล Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train โมเดล
    model = LinearRegression()
    model.fit(X_train, y_train)

    # ทำนายผล
    y_pred = model.predict(X_test)

    # พล็อตกราฟ
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)  # แสดงกราฟใน Streamlit
