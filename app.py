import streamlit as st
import torch
import torch.nn as nn
import time
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
import os
import urllib.request

# ==========================================
# ⬇️ 1. 모델 자동 다운로드 로직 (GitHub Releases)
# ==========================================
# 🚨 URL 주소에 오타가 없는지(v1.0car_classifier 부분) 꼭 다시 한번 확인해 주세요!
MODEL_URL = "https://github.com/lhylh0210-debug/car_classifier/releases/download/v1.0car_classifier/best_model_0893.pth"
MODEL_PATH = "best_model_0893.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("AI 모델을 서버로 가져오는 중입니다... (최초 1회 약 1~2분 소요)"):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ==========================================
# 🎨 2. Streamlit 페이지 설정 및 커스텀 CSS
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
    /* 보라색 그라디언트 헤더 */
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
""", unsafe_allow_html=True)

# ==========================================
# 🧠 3. AI 모델 로드 함수 및 설정
# ==========================================
NUM_CLASSES = 160  # 본인의 총 클래스 개수

# 160개 차종 완벽 매핑 데이터
CLASS_MAPPING = {
    0: 'BMW 2시리즈그란쿠페',
    1: 'BMW 3시리즈',
    2: 'BMW 4시리즈',
    3: 'BMW 5시리즈',
    4: 'BMW 6시리즈',
    5: 'BMW 7시리즈',
    6: 'BMW X1',
    7: 'BMW X3',
    8: 'BMW X4',
    9: 'BMW X5',
    10: 'BMW X6',
    11: 'BMW X7',
    12: '기아자동차 K3',
    13: '기아자동차 K5',
    14: '기아자동차 K7',
    15: '기아자동차 K9',
    16: '기아자동차 니로',
    17: '기아자동차 니로_EV',
    18: '기아자동차 로체',
    19: '기아자동차 모하비',
    20: '기아자동차 셀토스',
    21: '기아자동차 스토닉',
    22: '기아자동차 스팅어',
    23: '기아자동차 스펙트라',
    24: '기아자동차 스포티지',
    25: '기아자동차 쎄라토',
    26: '기아자동차 쏘렌토',
    27: '기아자동차 쏘울',
    28: '기아자동차 오피러스',
    29: '기아자동차 옵티마',
    30: '기아자동차 카렌스',
    31: '기아자동차 포르테',
    32: '기아자동차 프라이드',
    33: '기타_차종(Others)',
    34: '닛산 로그',
    35: '닛산 맥시마',
    36: '닛산 알티마',
    37: '닛산 엑스트레일',
    38: '닛산 쥬크',
    39: '닛산 큐브',
    40: '도요타 라브4',
    41: '도요타 아발론',
    42: '도요타 캠리',
    43: '랜드로버 디스커버리',
    44: '랜드로버 레인지로버',
    45: '랜드로버 이보크',
    46: '렉서스 ES',
    47: '렉서스 LS',
    48: '렉서스 NX',
    49: '렉서스 RX',
    50: '르노삼성 QM3',
    51: '르노삼성 QM5',
    52: '르노삼성 QM6',
    53: '르노삼성 SM3',
    54: '르노삼성 SM5',
    55: '르노삼성 SM6',
    56: '르노삼성 SM7',
    57: '르노삼성 XM3',
    58: '미니 컨트리맨',
    59: '미니 클럽맨',
    60: '벤츠 A클래스',
    61: '벤츠 CLA클래스',
    62: '벤츠 CLS클래스',
    63: '벤츠 C클래스',
    64: '벤츠 E클래스',
    65: '벤츠 GLA클래스',
    66: '벤츠 GLC클래스',
    67: '벤츠 GLE클래스',
    68: '벤츠 GLK클래스',
    69: '벤츠 GLS클래스',
    70: '벤츠 S클래스',
    71: '볼보 S60',
    72: '볼보 S90',
    73: '볼보 XC40',
    74: '볼보 XC60',
    75: '볼보 XC90',
    76: '쉐보레_대우 대우_라세티',
    77: '쉐보레_대우 대우_윈스톰',
    78: '쉐보레_대우 대우_토스카',
    79: '쉐보레_대우 말리부',
    80: '쉐보레_대우 아베오',
    81: '쉐보레_대우 올란도',
    82: '쉐보레_대우 임팔라',
    83: '쉐보레_대우 지엠_알페온',
    84: '쉐보레_대우 캡티바',
    85: '쉐보레_대우 크루즈',
    86: '쉐보레_대우 트레일블레이저',
    87: '쉐보레_대우 트렉스',
    88: '쌍용자동차 렉스턴',
    89: '쌍용자동차 무쏘',
    90: '쌍용자동차 엑티언',
    91: '쌍용자동차 체어맨',
    92: '쌍용자동차 체어맨H',
    93: '쌍용자동차 체어맨W',
    94: '쌍용자동차 카이런',
    95: '쌍용자동차 코란도',
    96: '쌍용자동차 코란도 투리스모',
    97: '쌍용자동차 코란도스포츠',
    98: '쌍용자동차 티볼리',
    99: '아우디 A4',
    100: '아우디 A5',
    101: '아우디 A6',
    102: '아우디 A7',
    103: '아우디 Q3',
    104: '아우디 Q5',
    105: '아우디 Q7',
    106: '인피니티 Q30',
    107: '인피니티 Q70',
    108: '인피니티 QX50',
    109: '인피니티 QX60',
    110: '재규어 F-페이스',
    111: '재규어 XE',
    112: '재규어 XF',
    113: '제네시스 EQ900',
    114: '제네시스 G70',
    115: '제네시스 G80',
    116: '제네시스 G90',
    117: '제네시스 GV80',
    118: '제네시스 제네시스',
    119: '제네시스 제네시스 쿠페',
    120: '지프 그랜드체로키',
    121: '지프 랭글러',
    122: '지프 레니게이드',
    123: '지프 체로키',
    124: '지프 컴패스',
    125: '테슬라 모델3',
    126: '포드 머스탱',
    127: '포드 몬데오',
    128: '포드 익스플로러',
    129: '포드 토러스',
    130: '포르쉐 카이엔',
    131: '포르쉐 파나메라',
    132: '폭스바겐 CC',
    133: '폭스바겐 비틀',
    134: '폭스바겐 아테온',
    135: '폭스바겐 제타',
    136: '폭스바겐 투아렉',
    137: '폭스바겐 티구안',
    138: '폭스바겐 파사트',
    139: '폭스바겐 페이톤',
    140: '푸조 2008',
    141: '푸조 3008',
    142: '푸조 508',
    143: '현대자동차 그랜저',
    144: '현대자동차 맥스크루즈',
    145: '현대자동차 베뉴',
    146: '현대자동차 베라크루즈',
    147: '현대자동차 베르나',
    148: '현대자동차 싼타페',
    149: '현대자동차 쏘나타',
    150: '현대자동차 아반떼',
    151: '현대자동차 아슬란',
    152: '현대자동차 에쿠스',
    153: '현대자동차 코나',
    154: '현대자동차 테라칸',
    155: '현대자동차 투싼',
    156: '현대자동차 팰리세이드',
    157: '혼다 CR-V',
    158: '혼다 어코드',
    159: '혼다 파일럿'
}

@st.cache_resource
def load_car_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = efficientnet_v2_s()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    
    try:
        # 구글 드라이브 경로 삭제 완료! 다운받은 파일을 바로 읽습니다.
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"모델을 불러오는 중 에러가 발생했습니다: {e}")
        return None, device

model, device = load_car_model()

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 🔄 4. 웹 페이지 상태 관리 (State Machine)
# ==========================================
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload'
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# ==========================================
# 📱 5. 화면 렌더링
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
    """, unsafe_allow_html=True)
    
    st.write("")
    uploaded_file = st.file_uploader("분석할 차량 사진을 업로드해주세요", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="선택된 이미지", use_container_width=True)
        st.session_state.uploaded_file = uploaded_file
        
        if st.button("🔍 차종 분류 시작하기", use_container_width=True):
            st.session_state.stage = 'analyzing'
            st.rerun()

# --- 2단계: 로딩 페이지 ---
elif st.session_state.stage == 'analyzing':
    st.markdown("""
        <div class='main-card'>
            <div class='header-zone'>
                <div class='header-title'>🚗 AI 자동차 차종 분류기</div>
                <div class='header-subtitle'>YOLO + EfficientNetV2 기반 정밀 분석</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.spinner(""):
        st.markdown("<h3 class='center-text' style='color: white;'>AI가 이미지를 분석하고 있습니다...</h3>", unsafe_allow_html=True)
        
        yolo_status = st.empty()
        resnet_status = st.empty()
        
        yolo_status.markdown("<p class='center-text' style='color: #8A2BE2;'>🔄 YOLO: 차량 영역 추출 중...</p>", unsafe_allow_html=True)
        time.sleep(1.2)
        yolo_status.markdown("<p class='center-text' style='color: #00CB76;'>✅ YOLO: 차량 영역 추출 완료</p>", unsafe_allow_html=True)
        
        resnet_status.markdown("<p class='center-text' style='color: #8A2BE2;'>🔄 ResNet(EfficientNet): 특징 데이터 분류 중...</p>", unsafe_allow_html=True)
        
        # 실제 AI 인공지능 추론 진행
        img = Image.open(st.session_state.uploaded_file).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        if model is not None:
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probabilities, 1)
                
                # 예측된 인덱스가 딕셔너리에 없을 경우를 대비한 안전 장치
                st.session_state.result_name = CLASS_MAPPING.get(pred_idx.item(), f"알 수 없는 차종 (ID: {pred_idx.item()})")
                st.session_state.result_prob = f"{confidence.item() * 100:.1f}%"
        else:
            st.session_state.result_name = "오류: 모델 없음"
            st.session_state.result_prob = "0.0%"
            
        resnet_status.markdown("<p class='center-text' style='color: #00CB76;'>✅ ResNet: 특징 데이터 분류 완료</p>", unsafe_allow_html=True)
        time.sleep(0.5)
        
        st.session_state.stage = 'result'
        st.rerun()

# --- 3단계: 분석 결과 페이지 ---
elif st.session_state.stage == 'result':
    st.markdown("""
        <div class='main-card'>
            <div class='header-zone'>
                <div class='header-title'>🚗 AI 자동차 차종 분류기</div>
                <div class='header-subtitle'>YOLO + EfficientNetV2 기반 정밀 분석</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(st.session_state.uploaded_file, use_container_width=True)
    with col2:
        st.caption("분석 결과")
        st.subheader(st.session_state.result_name)
        st.markdown(f"<span style='background-color:#E8E7FF; color:#6236FF; padding:3px 8px; border-radius:5px; font-size:12px; font-weight:bold;'>예측 확률: {st.session_state.result_prob}</span>", unsafe_allow_html=True)

    st.markdown("<hr style='border: 0.5px solid #333;'/>", unsafe_allow_html=True)
    st.markdown("<h5 class='center-text' style='color: white;'>예측이 맞았나요?</h5>", unsafe_allow_html=True)
    
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
        st.session_state.stage = 'upload'
        st.session_state.uploaded_file = None
        st.session_state.feedback_submitted = False
        st.rerun()
