import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10

# โหลด CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# ฟังก์ชันสำหรับโหลดโมเดล
def load_model():
    try:
        # พยายามโหลดโมเดลจากไฟล์ที่บันทึกไว้
        model = tf.keras.models.load_model('cifar10_model.h5')
    except:
        # ถ้าไม่พบไฟล์ ให้สร้างและเทรนโมเดลใหม่
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train / 255.0, y_train, epochs=10, validation_data=(x_test / 255.0, y_test), verbose=0)
        
        # บันทึกโมเดลลงในไฟล์
        model.save('cifar10_model.h5')
    
    return model

st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon=":guardsman:", layout="centered")
st.title("CNN Image Classifier with CIFAR-10")
st.markdown("""
    สวัสดีครับ! นี้คือแอปพลิเคชันที่ใช้โมเดล CNN (Convolutional Neural Network)
    เพื่อทำนายประเภทของภาพที่สุ่มจากชุดข้อมูล CIFAR-10
    กดปุ่ม **"สุ่มภาพใหม่"** เพื่อให้ระบบทำนายภาพและแสดงผลการทำนาย
""", unsafe_allow_html=True)

# ฟังก์ชันสุ่มภาพใหม่
def get_random_image():
    index = np.random.choice(len(x_test))
    image = x_test[index]
    return image, index

# กดปุ่มเพื่อสุ่มภาพใหม่และทำการทำนาย
if st.button("สุ่มภาพใหม่", key="random_image_button"):
    # โหลดโมเดล
    model = load_model()
    
    # สุ่มภาพใหม่
    img, idx = get_random_image()
    
    # แสดงภาพที่สุ่มมา
    st.subheader("ภาพที่สุ่มมา")
    st.image(img, width=300)
    
    # ป้อนข้อมูลให้โมเดลทำนาย
    img_array = np.expand_dims(img / 255.0, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    
    # แสดงผลการทำนาย
    st.write(f"### ผลการทำนาย:")
    st.markdown(f"**โมเดลทำนายว่าเป็น: {predicted_class}**", unsafe_allow_html=True)
    
    # ใช้สีตามประเภท
    color_map = {
        "airplane": "blue",
        "automobile": "green",
        "bird": "orange",
        "cat": "red",
        "deer": "brown",
        "dog": "purple",
        "frog": "yellow",
        "horse": "pink",
        "ship": "cyan",
        "truck": "magenta"
    }
    st.markdown(f'<h3 style="color:{color_map[predicted_class]}">สวยงามมาก! นี่คือ {predicted_class}</h3>', unsafe_allow_html=True)
