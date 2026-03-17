import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import math

# 1. 強健的導入方式 (避免之前遇到的 AttributeError)
try:
    from mediapipe.python.solutions import hands as mp_hands
except ImportError:
    st.error("Mediapipe 載入失敗，請確認 requirements.txt 包含 opencv-python-headless")

# 初始化 MediaPipe Hands
@st.cache_resource
def load_hands_model():
    return mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=2, 
        min_detection_confidence=0.5
    )

hands = load_hands_model()

def overlay_nail(background, nail_img, finger_tip, finger_mcp):
    """
    將美甲圖片旋轉、縮放並覆蓋到指尖
    """
    # 計算手指的角度
    dx = finger_tip.x - finger_mcp.x
    dy = finger_tip.y - finger_mcp.y
    angle = math.degrees(math.atan2(dy, dx)) + 90  # 修正垂直角度
    
    # 計算指甲大小 (根據手指長度縮放)
    dist = math.sqrt(dx**2 + dy**2)
    h, w = background.shape[:2]
    nail_size = int(dist * h * 0.4) # 比例係數，可依需求調整
    if nail_size < 10: nail_size = 10

    # 縮放與旋轉美甲貼圖
    nail_resized = cv2.resize(nail_img, (nail_size, nail_size))
    center = (nail_size // 2, nail_size // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    nail_rotated = cv2.warpAffine(nail_resized, M, (nail_size, nail_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # 計算貼上位置 (指尖座標轉換為像素)
    px = int(finger_tip.x * w)
    py = int(finger_tip.y * h)
    
    # 建立遮罩進行 Alpha 合成 (處理去背 PNG)
    y1, y2 = max(0, py - nail_size//2), min(h, py + nail_size//2)
    x1, x2 = max(0, px - nail_size//2), min(w, px + nail_size//2)
    
    # 剪裁要貼上的區域
    nail_part = nail_rotated[0:(y2-y1), 0:(x2-x1)]
    
    if nail_part.shape[2] == 4: # 如果有 Alpha 通道
        alpha_nail = nail_part[:, :, 3] / 255.0
        alpha_bg = 1.0 - alpha_nail
        for c in range(0, 3):
            background[y1:y2, x1:x2, c] = (alpha_nail * nail_part[:, :, c] +
                                          alpha_bg * background[y1:y2, x1:x2, c])
    return background

# --- Streamlit UI ---
st.set_page_config(page_title="AI Nail Designer", layout="centered")
st.title("💅 AI 虛擬美甲設計師")

# 側邊欄：選擇美甲貼圖
st.sidebar.header("美甲款式")
nail_option = st.sidebar.selectbox("選擇款式", ["法式簡約", "自定義圖片"])

if nail_option == "自定義圖片":
    nail_file = st.sidebar.file_uploader("上傳去背的指甲貼圖 (PNG)", type=["png"])
else:
    # 這裡可以用一張網路上的範例 PNG 或你的 assets 檔案
    st.sidebar.info("請上傳一張去背的指甲 PNG 圖片來開始")
    nail_file = None

# 主畫面：上傳手部照片
hand_file = st.file_uploader("1. 上傳你的手部照片", type=["jpg", "jpeg", "png"])

if hand_file and nail_file:
    # 讀取手部圖片
    hand_img = Image.open(hand_file).convert("RGB")
    hand_array = np.array(hand_img)
    
    # 讀取美甲圖片 (包含 Alpha 通道)
    nail_img = Image.open(nail_file).convert("RGBA")
    nail_array = np.array(nail_img)
    
    # 偵測手部關鍵點
    results = hands.process(cv2.cvtColor(hand_array, cv2.COLOR_RGB2BGR))
    
    if results.multi_hand_landmarks:
        output_img = hand_array.copy()
        # 每一根手指的尖端 (Landmarks: 4, 8, 12, 16, 20)
        tips = [4, 8, 12, 16, 20]
        mcps = [3, 7, 11, 15, 19] # 對應的指節，用來算角度
        
        for hand_lms in results.multi_hand_landmarks:
            for tip_idx, mcp_idx in zip(tips, mcps):
                tip = hand_lms.landmark[tip_idx]
                mcp = hand_lms.landmark[mcp_idx]
                output_img = overlay_nail(output_img, nail_array, tip, mcp)
        
        st.image(output_img, caption="美甲合成效果", use_column_width=True)
        st.success("合成完成！")
    else:
        st.error("未偵測到手部，請確保手指清晰可見。")
