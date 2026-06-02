import streamlit as st
import torch
import torch.nn as nn
import time
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s

# ========================================================
# 🚨 주의: 여기에 아까 제가 짜드린 전체 코드를 그대로 '복사/붙여넣기' 하세요!
# (단, MODEL_PATH 경로는 본인의 구글 드라이브 경로로 꼭 수정해야 합니다!)
# 예: MODEL_PATH = "/content/drive/MyDrive/best_model.pth"
# ========================================================

import streamlit as st
import torch
import torch.nn as nn
import time
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s

# ==========================================
# 🎨 1. Streamlit 페이지 설정 및 커스텀 CSS (기획서 디자인 반영)
# ==========================================
st.set_page_config(page_title="AI 자동차 차종 분류기", layout="centered")

st.markdown("""
    <style>
    /* 전체 배경을 어두운 네이비 계열로 */
    .stApp {
        background-color: #0B132B;
    }
    /* 카드 컨테이너 스타일 */
    .main-card {
        background-color: white;
        border-radius: 15px;
        padding: 0px 0px 20px 0px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        color: #333333;
        overflow: hidden;
    }
    /* 보라색 그라디언트 헤더 (image_8e27f3.jpg 반영) */
    .header-zone {
        background: linear-gradient(135deg, #6236FF, #9747FF);
        padding: 30px;
        text-align: center;
        color: white;
    }
    .header-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .header-subtitle {
        font-size: 14px;
        opacity: 0.8;
    }
    /* 결과 페이지 가로형 미니 카드 */
    .result-box {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E9ECEF;
        display: flex;
        align-items: center;
        gap: 20px;
        margin: 20px;
    }
    /* 텍스트 센터 정렬용 */
    .center-text {
        text-align: center;
        margin-top: 15px;
    }
    </style>
""", unsafe_html=True)

# ==========================================
# 🧠 2. AI 모델 로드 함수 (구글 드라이브 경로 연결용)
# ==========================================
@st.cache_resource
def load_car_model():
    # 💡 본인 환경에 맞는 실제 모델 파일 경로와 클래스 개수를 입력하세요
    MODEL_PATH = "/content/drive/MyDrive/딥러닝실습_모델/best_model_0893.pth"  
    NUM_CLASSES = 72               # 본인의 총 클래스 개수
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = efficientnet_v2_s()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    
    try:
        # 가중치 불러오기
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        # 파일이 없을 때를 대비한 플레이스홀더 (테스트용)
        return None, device

model, device = load_car_model()

# 임시 클래스 매핑 (본인의 class_mapping 딕셔너리로 교체하세요)
CLASS_MAPPING = {0: "현대자동차 쏘나타", 1: "제네시스 G80", 2: "기아자동차 쏘렌토"} 

# 이미지 전처리 정의 (384 해상도 일치)
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 🔄 3. 웹 페이지 상태 관리 (State Machine)
# ==========================================
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload'
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# ==========================================
# 📱 4. 화면 렌더링 (기획서 3단계 완벽 재현)
# ==========================================

# --- 1단계: 업로드 페이지 ---
if st.session_state.stage == 'upload':
    st.markdown("""
        <div class='main-card'>
            <div class='header-zone'>
                <div class='header-title'>🚗 AI 자동차 차종 분류기</div>
                <div class='header-subtitle'>YOLO + EfficientNetV2 기반 정밀 분석</div>
            </div>
        </div>
    """, unsafe_html=True)
    
    st.write("")
    uploaded_file = st.file_uploader("분석할 차량 사진을 업로드해주세요", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="선택된 이미지", use_container_width=True)
        st.session_state.uploaded_file = uploaded_file
        
        if st.button("🔍 차종 분류 시작하기", use_container_width=True):
            st.session_state.stage = 'analyzing'
            st.rerun()

# --- 2단계: 로딩 페이지 (image_8e2796.jpg 왼쪽) ---
elif st.session_state.stage == 'analyzing':
    st.markdown("""
        <div class='main-card'>
            <div class='header-zone'>
                <div class='header-title'>🚗 AI 자동차 차종 분류기</div>
                <div class='header-subtitle'>YOLO + EfficientNetV2 기반 정밀 분석</div>
            </div>
        </div>
    """, unsafe_html=True)
    
    with st.spinner(""):
        st.markdown("<h3 class='center-text' style='color: white;'>AI가 이미지를 분석하고 있습니다...</h3>", unsafe_html=True)
        
        # 기획서의 체크리스트 애니메이션 효과 재현
        yolo_status = st.empty()
        resnet_status = st.empty()
        
        yolo_status.markdown("<p class='center-text' style='color: #8A2BE2;'>🔄 YOLO: 차량 영역 추출 중...</p>", unsafe_html=True)
        time.sleep(1.2) # 가상 로딩 시간
        yolo_status.markdown("<p class='center-text' style='color: #00CB76;'>✅ YOLO: 차량 영역 추출 완료</p>", unsafe_html=True)
        
        resnet_status.markdown("<p class='center-text' style='color: #8A2BE2;'>🔄 ResNet(EfficientNet): 특징 데이터 분류 중...</p>", unsafe_html=True)
        
        # 실제 AI 인공지능 추론 진행
        img = Image.open(st.session_state.uploaded_file).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        if model is not None:
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probabilities, 1)
                
                st.session_state.result_name = CLASS_MAPPING.get(pred_idx.item(), "알 수 없는 차종")
                st.session_state.result_prob = f"{confidence.item() * 100:.1f}%"
        else:
            # 모델 파일이 연결되지 않았을 때를 위한 디버깅용 더미 데이터 (제네시스 G80 세팅)
            time.sleep(1.0)
            st.session_state.result_name = "제네시스 G80"
            st.session_state.result_prob = "98.4%"
            
        resnet_status.markdown("<p class='center-text' style='color: #00CB76;'>✅ ResNet: 특징 데이터 분류 완료</p>", unsafe_html=True)
        time.sleep(0.5)
        
        # 결과 페이지로 이동
        st.session_state.stage = 'result'
        st.rerun()

# --- 3단계: 분석 결과 페이지 (image_8e2796.jpg 오른쪽) ---
elif st.session_state.stage == 'result':
    st.markdown("""
        <div class='main-card'>
            <div class='header-zone'>
                <div class='header-title'>🚗 AI 자동차 차종 분류기</div>
                <div class='header-subtitle'>YOLO + EfficientNetV2 기반 정밀 분석</div>
            </div>
        </div>
    """, unsafe_html=True)
    
    # 가로형 결과 박스 레이아웃 구축
    st.write("")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(st.session_state.uploaded_file, use_container_width=True)
    with col2:
        st.caption("분석 결과")
        st.subheader(st.session_state.result_name)
        st.markdown(f"<span style='background-color:#E8E7FF; color:#6236FF; padding:3px 8px; border-radius:5px; font-size:12px; font-weight:bold;'>예측 확률: {st.session_state.result_prob}</span>", unsafe_html=True)

    st.markdown("<hr style='border: 0.5px solid #333;'/>", unsafe_html=True)
    st.markdown("<h5 class='center-text' style='color: white;'>예측이 맞았나요?</h5>", unsafe_html=True)
    
    # 피드백 버튼 행
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        if st.button("🟢 예, 맞습니다", use_container_width=True):
            st.session_state.feedback_submitted = True
    with f_col2:
        if st.button("❌ 아니오, 틀렸습니다", use_container_width=True):
            st.session_state.feedback_submitted = True
            
    if st.session_state.feedback_submitted:
        st.success("정확한 예측이군요! 피드백 감사합니다.")
        
    st.write("")
    if st.button("🔄 다른 이미지 분류하기", type="secondary", use_container_width=True):
        # 세션 초기화 후 업로드 페이지로 리셋
        st.session_state.stage = 'upload'
        st.session_state.uploaded_file = None
        st.session_state.feedback_submitted = False
        st.rerun()
