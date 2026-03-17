import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# 初始化 Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

st.title("💅 AI 虛擬美甲設計師")
st.write("上傳你的手部照片，並選擇喜歡的彩繪圖案！")

# 1. 上傳手部照片
hand_file = st.file_uploader("上傳手部照片", type=["jpg", "png", "jpeg"])

# 2. 選擇或上傳美甲樣式
nail_style = st.sidebar.radio("選擇美甲樣式", ["粉紅碎花", "極簡法式", "自定義上傳"])
if nail_style == "自定義上傳":
    style_file = st.sidebar.file_uploader("上傳美甲貼圖 (建議去背 PNG)", type=["png"])
else:
    # 這裡放你預設的圖片路徑
    # style_file = "assets/flower.png"
    pass

if hand_file:
    # 轉換圖片格式
    image = Image.open(hand_file)
    img_array = np.array(image)
    results = hands.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    if results.multi_hand_landmarks:
        st.success("偵測到手部！正在套用美甲...")
        
        # 這裡實作影像合成邏輯
        # 遍歷 landmarks，找到指尖位置，並用 cv2.warpAffine 或 seamlessClone 覆蓋圖案
        # ... (影像處理邏輯) ...
        
        st.image(img_array, caption="生成結果", use_column_width=True)
    else:
        st.warning("未能偵測到清晰的手部，請換張照片試試。")
