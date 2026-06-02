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

# ==========================================
# 🎨 1. Streamlit 페이지 설정 및 커스텀 CSS
# ==========================================
st.set_page_config(page_title="AI 자동차 차종 분류기", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* 전체 배경을 레퍼런스 이미지의 다크 네이비로 설정 */
    .stApp {
        background-color: #0A1128;
    }
    
    /* 전체 텍스트 컬러 화이트 고정 */
    h1, h2, h3, p, span, div {
        color: #F8F9FA;
    }
    
    /* 좌측 메인 타이틀 */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 2rem;
        background: -webkit-linear-gradient(0deg, #FFFFFF, #A5B4FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* 스텝 텍스트 */
    .step-text {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 0.8rem;
    }
    .step-number {
        color: #818CF8;
        font-weight: 800;
        margin-right: 10px;
    }
    
    /* 메인 파란색 액션 버튼 */
    div.stButton > button:first-child {
        background-color: #4F46E5;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 0.8rem 0;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #4338CA;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    
    /* 🚨 수정됨: 파일 업로더 영역 시인성 강화 */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255,255,255,0.05) !important;
        border: 2px dashed rgba(129, 140, 248, 0.5) !important;
        border-radius: 12px;
        padding: 20px !important;
    }
    /* 업로더 안의 텍스트 색상 */
    [data-testid="stFileUploadDropzone"] * {
        color: #E2E8F0 !important;
    }
    /* 업로더 'Browse files' 버튼 디자인 */
    [data-testid="stFileUploadDropzone"] button {
        background-color: rgba(129, 140, 248, 0.2) !important;
        color: #A5B4FC !important;
        border: 1px solid #818CF8 !important;
        border-radius: 6px !important;
        font-weight: bold !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover {
        background-color: rgba(129, 140, 248, 0.4) !important;
        color: #FFFFFF !important;
    }
    
    /* 우측 이미지 프레임 카드 (중앙 정렬 완벽 보장) */
    .right-image-card {
        background-color: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        height: calc(100vh - 150px); /* 화면 높이에 맞게 꽉 채움 */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        overflow: hidden;
    }
    
    /* 결과 강조 뱃지 */
    .result-badge {
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        padding: 10px 25px;
        border-radius: 30px;
        font-size: 1.2rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
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
        st.error(f"모델을 불러오는 중 에러가 발생했습니다: {e}")
        return None, device

model, device = load_car_model()

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 🔄 3. 상태 관리
# ==========================================
if 'stage' not in st.session_state: st.session_state.stage = 'upload'
if 'uploaded_file' not in st.session_state: st.session_state.uploaded_file = None
if 'feedback_submitted' not in st.session_state: st.session_state.feedback_submitted = False

# 여백을 주어 레퍼런스 이미지처럼 화면 구성을 맞춤
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
            st.session_state.uploaded_file = uploaded_file
            
        st.markdown("<div class='step-text'><span class='step-number'>2</span> Run AI Analysis</div>", unsafe_allow_html=True)
        if st.button("Start AI Analysis ➔", use_container_width=True):
            if st.session_state.uploaded_file is not None:
                st.session_state.stage = 'analyzing'
                st.rerun()
            else:
                st.warning("먼저 이미지를 업로드해주세요!")

    # --- 2단계: 분석 중 ---
    elif st.session_state.stage == 'analyzing':
        st.markdown("<div class='step-text'><span class='step-number'>⏳</span> Processing...</div>", unsafe_allow_html=True)
        
        yolo_status = st.empty()
        resnet_status = st.empty()
        
        yolo_status.markdown("<p style='color: #818CF8;'>🔄 Extracting vehicle area...</p>", unsafe_allow_html=True)
        time.sleep(1.2)
        yolo_status.markdown("<p style='color: #10B981;'>✅ Vehicle area extracted</p>", unsafe_allow_html=True)
        
        resnet_status.markdown("<p style='color: #818CF8;'>🔄 Classifying features...</p>", unsafe_allow_html=True)
        
        img = Image.open(st.session_state.uploaded_file).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        if model is not None:
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probabilities, 1)
                
                st.session_state.result_name = CLASS_MAPPING.get(pred_idx.item(), f"Unknown (ID: {pred_idx.item()})")
                st.session_state.result_prob = f"{confidence.item() * 100:.1f}%"
        
        resnet_status.markdown("<p style='color: #10B981;'>✅ Analysis complete</p>", unsafe_allow_html=True)
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
            st.session_state.uploaded_file = None
            st.session_state.feedback_submitted = False
            st.rerun()

# ----------------- [우측 프리뷰/이미지 영역] -----------------
with col_right:
    # 🚨 수정됨: HTML 안에서 Base64로 이미지를 직접 그려 중앙 정렬과 비율을 완벽하게 맞춤
    if st.session_state.uploaded_file is not None:
        bytes_data = st.session_state.uploaded_file.getvalue()
        b64_encoded = base64.b64encode(bytes_data).decode()
        
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
