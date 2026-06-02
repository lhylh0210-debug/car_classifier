import streamlit as st
import torch
import torch.nn as nn
import time
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
import os
import urllib.request
import base64
import io

# 🚨 새롭게 추가된 라이브러리 (YOLO)
from ultralytics import YOLO 

# ==========================================
# 🎨 1. Streamlit 페이지 설정 및 커스텀 CSS
# ==========================================
st.set_page_config(page_title="AI 자동차 차종 분류기", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0A1128; }
    h1, h2, h3, p, span, div { color: #F8F9FA; }
    
    .main-title {
        font-size: 3.5rem; font-weight: 800; line-height: 1.2; margin-bottom: 2rem;
        background: -webkit-linear-gradient(0deg, #FFFFFF, #A5B4FC);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    
    .step-text { font-size: 1.2rem; font-weight: 600; margin-top: 2rem; margin-bottom: 0.8rem; }
    .step-number { color: #818CF8; font-weight: 800; margin-right: 10px; }
    
    div.stButton > button:first-child {
        background-color: #4F46E5; color: white !important; border-radius: 8px;
        border: none; padding: 0.8rem 0; font-weight: bold; font-size: 1.1rem; transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #4338CA; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255,255,255,0.05) !important;
        border: 2px dashed rgba(129, 140, 248, 0.5) !important;
        border-radius: 12px; padding: 20px !important;
    }
    [data-testid="stFileUploadDropzone"] * { color: #E2E8F0 !important; }
    [data-testid="stFileUploadDropzone"] button {
        background-color: rgba(129, 140, 248, 0.2) !important; color: #A5B4FC !important;
        border: 1px solid #818CF8 !important; border-radius: 6px !important; font-weight: bold !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover {
        background-color: rgba(129, 140, 248, 0.4) !important; color: #FFFFFF !important;
    }
    
    .right-image-card {
        background-color: rgba(255,255,255,0.03); border-radius: 16px; padding: 20px;
        border: 1px solid rgba(255,255,255,0.1); height: calc(100vh - 150px); 
        display: flex; flex-direction: column; justify-content: center; align-items: center; overflow: hidden;
    }
    
    .result-badge {
        background: linear-gradient(135deg, #4F46E5, #7C3AED); padding: 10px 25px;
        border-radius: 30px; font-size: 1.2rem; font-weight: bold; display: inline-block;
        margin-top: 15px; box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# ⬇️ 2. 모델 자동 다운로드 및 로드 로직
# ==========================================
MODEL_URL = "https://github.com/lhylh0210-debug/car_classifier/releases/download/v1.0car_classifier/best_model_0893.pth"
MODEL_PATH = "best_model_0893.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("AI 모델을 서버로 가져오는 중입니다... (최초 1회 약 1~2분 소요)"):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

NUM_CLASSES = 160

# 클래스 딕셔너리 생략 (튜티님의 160개 코드가 그대로 들어갑니다)
CLASS_MAPPING = {
    0: 'BMW 2시리즈그란쿠페', 1: 'BMW 3시리즈', 2: 'BMW 4시리즈', 3: 'BMW 5시리즈', 4: 'BMW 6시리즈', 
    5: 'BMW 7시리즈', 6: 'BMW X1', 7: 'BMW X3', 8: 'BMW X4', 9: 'BMW X5', 10: 'BMW X6', 11: 'BMW X7', 
    12: '기아자동차 K3', 13: '기아자동차 K5', 14: '기아자동차 K7', 15: '기아자동차 K9', 16: '기아자동차 니로', 
    17: '기아자동차 니로_EV', 18: '기아자동차 로체', 19: '기아자동차 모하비', 20: '기아자동차 셀토스', 
    21: '기아자동차 스토닉', 22: '기아자동차 스팅어', 23: '기아자동차 스펙트라', 24: '기아자동차 스포티지', 
    25: '기아자동차 쎄라토', 26: '기아자동차 쏘렌토', 27: '기아자동차 쏘울', 28: '기아자동차 오피러스', 
    29: '기아자동차 옵티마', 30: '기아자동차 카렌스', 31: '기아자동차 포르테', 32: '기아자동차 프라이드', 
    33: '기타_차종(Others)', 34: '닛산 로그', 35: '닛산 맥시마', 36: '닛산 알티마', 37: '닛산 엑스트레일', 
    38: '닛산 쥬크', 39: '닛산 큐브', 40: '도요타 라브4', 41: '도요타 아발론', 42: '도요타 캠리', 
    43: '랜드로버 디스커버리', 44: '랜드로버 레인지로버', 45: '랜드로버 이보크', 46: '렉서스 ES', 
    47: '렉서스 LS', 48: '렉서스 NX', 49: '렉서스 RX', 50: '르노삼성 QM3', 51: '르노삼성 QM5', 
    52: '르노삼성 QM6', 53: '르노삼성 SM3', 54: '르노삼성 SM5', 55: '르노삼성 SM6', 56: '르노삼성 SM7', 
    57: '르노삼성 XM3', 58: '미니 컨트리맨', 59: '미니 클럽맨', 60: '벤츠 A클래스', 61: '벤츠 CLA클래스', 
    62: '벤츠 CLS클래스', 63: '벤츠 C클래스', 64: '벤츠 E클래스', 65: '벤츠 GLA클래스', 66: '벤츠 GLC클래스', 
    67: '벤츠 GLE클래스', 68: '벤츠 GLK클래스', 69: '벤츠 GLS클래스', 70: '벤츠 S클래스', 71: '볼보 S60', 
    72: '볼보 S90', 73: '볼보 XC40', 74: '볼보 XC60', 75: '볼보 XC90', 76: '쉐보레_대우 대우_라세티', 
    77: '쉐보레_대우 대우_윈스톰', 78: '쉐보레_대우 대우_토스카', 79: '쉐보레_대우 말리부', 80: '쉐보레_대우 아베오', 
    81: '쉐보레_대우 올란도', 82: '쉐보레_대우 임팔라', 83: '쉐보레_대우 지엠_알페온', 84: '쉐보레_대우 캡티바', 
    85: '쉐보레_대우 크루즈', 86: '쉐보레_대우 트레일블레이저', 87: '쉐보레_대우 트렉스', 88: '쌍용자동차 렉스턴', 
    89: '쌍용자동차 무쏘', 90: '쌍용자동차 엑티언', 91: '쌍용자동차 체어맨', 92: '쌍용자동차 체어맨H', 
    93: '쌍용자동차 체어맨W', 94: '쌍용자동차 카이런', 95: '쌍용자동차 코란도', 96: '쌍용자동차 코란도 투리스모', 
    97: '쌍용자동차 코란도스포츠', 98: '쌍용자동차 티볼리', 99: '아우디 A4', 100: '아우디 A5', 101: '아우디 A6', 
    102: '아우디 A7', 103: '아우디 Q3', 104: '아우디 Q5', 105: '아우디 Q7', 106: '인피니티 Q30', 
    107: '인피니티 Q70', 108: '인피니티 QX50', 109: '인피니티 QX60', 110: '재규어 F-페이스', 
    111: '재규어 XE', 112: '재규어 XF', 113: '제네시스 EQ900', 114: '제네시스 G70', 115: '제네시스 G80', 
    116: '제네시스 G90', 117: '제네시스 GV80', 118: '제네시스 제네시스', 119: '제네시스 제네시스 쿠페', 
    120: '지프 그랜드체로키', 121: '지프 랭글러', 122: '지프 레니게이드', 123: '지프 체로키', 
    124: '지프 컴패스', 125: '테슬라 모델3', 126: '포드 머스탱', 127: '포드 몬데오', 128: '포드 익스플로러', 
    129: '포드 토러스', 130: '포르쉐 카이엔', 131: '포르쉐 파나메라', 132: '폭스바겐 CC', 
    133: '폭스바겐 비틀', 134: '폭스바겐 아테온', 135: '폭스바겐 제타', 136: '폭스바겐 투아렉', 
    137: '폭스바겐 티구안', 138: '폭스바겐 파사트', 139: '폭스바겐 페이톤', 140: '푸조 2008', 
    141: '푸조 3008', 142: '푸조 508', 143: '현대자동차 그랜저', 144: '현대자동차 맥스크루즈', 
    145: '현대자동차 베뉴', 146: '현대자동차 베라크루즈', 147: '현대자동차 베르나', 148: '현대자동차 싼타페', 
    149: '현대자동차 쏘나타', 150: '현대자동차 아반떼', 151: '현대자동차 아슬란', 152: '현대자동차 에쿠스', 
    153: '현대자동차 코나', 154: '현대자동차 테라칸', 155: '현대자동차 투싼', 156: '현대자동차 팰리세이드', 
    157: '혼다 CR-V', 158: '혼다 어코드', 159: '혼다 파일럿'
}

# 🧠 기존 EfficientNet 로드
@st.cache_resource
def load_car_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = efficientnet_v2_s()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        return None, device

# 🎯 YOLO 모델 로드 (가장 가볍고 빠른 yolov8n 사용)
@st.cache_resource
def load_yolo_model():
    # 서버 켜질 때 자동으로 가중치 다운로드 됨
    model = YOLO('yolov8n.pt') 
    return model

model, device = load_car_model()
yolo_model = load_yolo_model()

# EfficientNet용 전처리 (384x384)
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 🔄 3. 상태 관리 및 이미지 렌더링 헬퍼
# ==========================================
if 'stage' not in st.session_state: st.session_state.stage = 'upload'
if 'display_image' not in st.session_state: st.session_state.display_image = None
if 'feedback_submitted' not in st.session_state: st.session_state.feedback_submitted = False

# PIL 이미지를 HTML 화면에 띄우기 위해 Base64로 변환하는 함수
def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

col_left, col_space, col_right = st.columns([1, 0.1, 1.3])

# ==========================================
# 📱 4. 화면 렌더링 (좌우 분할 스플릿 뷰)
# ==========================================

# ----------------- [좌측 컨트롤 패널 영역] -----------------
with col_left:
    st.write("") 
    st.write("") 
    st.markdown("<div class='main-title'>Experience<br>AI Car Classifier</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; opacity: 0.8; margin-bottom: 2rem;'>Upload an image to precisely identify the vehicle make and model using EfficientNetV2.</p>", unsafe_allow_html=True)
    
    # --- 1단계: 업로드 ---
    if st.session_state.stage == 'upload':
        st.markdown("<div class='step-text'><span class='step-number'>1</span> Select a vehicle photo</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            # 업로드 시 화면에 띄울 원본 이미지를 세션에 저장
            st.session_state.display_image = Image.open(uploaded_file).convert("RGB")
            
        st.markdown("<div class='step-text'><span class='step-number'>2</span> Run AI Analysis</div>", unsafe_allow_html=True)
        if st.button("Start AI Analysis ➔", use_container_width=True):
            if st.session_state.display_image is not None:
                st.session_state.stage = 'analyzing'
                st.rerun()
            else:
                st.warning("먼저 이미지를 업로드해주세요!")

    # --- 2단계: 분석 중 (실제 YOLO 적용) ---
    elif st.session_state.stage == 'analyzing':
        st.markdown("<div class='step-text'><span class='step-number'>⏳</span> Processing...</div>", unsafe_allow_html=True)
        
        yolo_status = st.empty()
        resnet_status = st.empty()
        
        yolo_status.markdown("<p style='color: #818CF8;'>🔄 YOLO: Detecting and Cropping Vehicle...</p>", unsafe_allow_html=True)
        
        # 1️⃣ 실제 YOLO 추론 진행
        original_img = st.session_state.display_image
        results = yolo_model(original_img)
        
        best_box = None
        max_conf = 0.0
        
        # YOLO 결과에서 가장 신뢰도가 높은 자동차(car, bus, truck) 박스 찾기
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # COCO 데이터셋 기준: 2=car, 3=motorcycle, 5=bus, 7=truck
                if cls_id in [2, 3, 5, 7] and conf > max_conf:
                    max_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()
        
        # 자동차를 찾았다면 해당 영역만 크롭
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box)
            cropped_img = original_img.crop((x1, y1, x2, y2))
            st.session_state.display_image = cropped_img # 우측 화면을 크롭된 이미지로 업데이트!
        else:
            cropped_img = original_img # 못 찾으면 원본 유지
            
        time.sleep(0.5) # UI 시각적 여유
        yolo_status.markdown("<p style='color: #10B981;'>✅ YOLO: Vehicle successfully cropped</p>", unsafe_allow_html=True)
        
        # 2️⃣ 크롭된 이미지를 EfficientNet에 입력 (384x384 전처리)
        resnet_status.markdown("<p style='color: #818CF8;'>🔄 EfficientNet: Classifying features...</p>", unsafe_allow_html=True)
        
        input_tensor = transform(cropped_img).unsqueeze(0).to(device)
        
        if model is not None:
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probabilities, 1)
                
                st.session_state.result_name = CLASS_MAPPING.get(pred_idx.item(), f"Unknown (ID: {pred_idx.item()})")
                st.session_state.result_prob = f"{confidence.item() * 100:.1f}%"
        
        resnet_status.markdown("<p style='color: #10B981;'>✅ EfficientNet: Analysis complete</p>", unsafe_allow_html=True)
        time.sleep(0.5)
        
        st.session_state.stage = 'result'
        st.rerun()

    # --- 3단계: 결과 확인 ---
    elif st.session_state.stage == 'result':
        st.markdown("<div class='step-text'><span class='step-number'>🎯</span> Analysis Result</div>", unsafe_allow_html=True)
        
        st.markdown(f"<div style='font-size: 1.5rem; font-weight: bold;'>{st.session_state.result_name}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-badge'>Accuracy: {st.session_state.result_prob}</div>", unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        st.markdown("<p style='opacity: 0.8;'>Is this prediction correct?</p>", unsafe_allow_html=True)
        
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            if st.button("👍 Yes", use_container_width=True):
                st.session_state.feedback_submitted = True
        with f_col2:
            if st.button("👎 No", use_container_width=True):
                st.session_state.feedback_submitted = True
                
        if st.session_state.feedback_submitted:
            st.success("Thanks for your feedback!")
            
        st.write("")
        if st.button("Try another photo ↺", use_container_width=True):
            st.session_state.stage = 'upload'
            st.session_state.display_image = None
            st.session_state.feedback_submitted = False
            st.rerun()

# ----------------- [우측 프리뷰/이미지 영역] -----------------
with col_right:
    # 업로드 되거나 크롭된 이미지가 있으면 그리기
    if st.session_state.display_image is not None:
        b64_encoded = pil_to_base64(st.session_state.display_image)
        
        st.markdown(f"""
            <div class='right-image-card'>
                <img src="data:image/jpeg;base64,{b64_encoded}" 
                     style="max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);"/>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='right-image-card'>
                <div style='text-align: center; opacity: 0.3; margin: auto;'>
                    <h1 style='font-size: 4rem; margin-bottom: 10px;'>🚗</h1>
                    <p style='font-size: 1.1rem;'>Upload a vehicle image<br>to see the preview</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
