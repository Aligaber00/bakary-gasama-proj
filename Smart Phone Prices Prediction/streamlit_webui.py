import streamlit as st
import numpy as np
import pandas as pd
import joblib
st.set_page_config(
    page_title="Smart Phone Price Predictor",
    page_icon="📱",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Title */
.title-text {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    color: #00f5ff;
    text-shadow: 0 0 20px rgba(0,245,255,0.6);
}

/* Subtitle */
.subtitle-text {
    text-align: center;
    font-size: 1.2rem;
    color: #d1d1d1;
    margin-bottom: 30px;
}

/* Card (Glassmorphism) */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Input labels */
label {
    color: #00f5ff !important;
    font-weight: 600;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #00f5ff, #00c3ff);
    color: black;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 25px;
    border: none;
    box-shadow: 0 0 20px rgba(0,245,255,0.6);
    transition: 0.3s ease-in-out;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 35px rgba(0,245,255,0.9);
}

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141e30, #243b55);
}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="title-text">📱 Smart Phone Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">AI-powered smartphone value estimation</div>', unsafe_allow_html=True)


brand_encoder = joblib.load("brand_target_encoder.pkl")
models = {
    "Voting Classifier": joblib.load("VotingClassifierSoft.pkl"),
    "Gradient Boosting": joblib.load("gradient_boosting_smote.pkl"),
    "Random Forest": joblib.load("random_forest_smote.pkl"),
    "Logistic Regression": joblib.load("logistic_regression_smote.pkl"),
    "SVM": joblib.load("svm_smote.pkl"),
    "KNN": joblib.load("knn_smote.pkl")
}
model_name = st.selectbox("Choose Model (Voting Classifier Recommended)", list(models.keys()))
model = models[model_name]

features = [
    'rating',
    '5G',
    'Vo5G',
    'NFC',
    'Clock_Speed_GHz',
    'RAM Size GB',
    'Storage Size GB',
    'RAM Tier',
    'fast_charging_power',
    'Screen_Size',
    'Resolution_Width',
    'Resolution_Height',
    'Refresh_Rate',
    'primary_rear_camera_mp',
    'num_rear_cameras',
    'primary_front_camera_mp',
    'memory_card_support',
    'os_version',
    'memory_card_size_gb',
    'brand_encoded',
    'Notch_Type_Dual Punch Hole',
    'Notch_Type_Large Notch',
    'Notch_Type_No Notch',
    'Notch_Type_Punch Hole',
    'Notch_Type_Small Notch',
    'Notch_Type_Water Drop Notch',
    'os_name_Android',
    'os_name_EMUI',
    'os_name_HarmonyOS',
    'os_name_KAI OS',
    'os_name_Pragati OS',
    'os_name_iOS',
    'Processor_Brand_Bionic',
    'Processor_Brand_Dimensity',
    'Processor_Brand_Exynos',
    'Processor_Brand_Google Tensor',
    'Processor_Brand_Helio',
    'Processor_Brand_Kirin',
    'Processor_Brand_Other',
    'Processor_Brand_Snapdragon',
    'Processor_Brand_Unisoc'
]

brands = [
    "Samsung", "Xiaomi", "Vivo", "Oppo", "Realme", "Oneplus", "Apple",
    "Motorola", "Poco", "Iqoo", "Tecno", "Infinix", "Nokia",
    "Realme narzo", "Huawei", "Google", "Honor", "Motorola edge",
    "Itel", "Sony", "Asus", "Nothing", "Nubia", "Lava", "Jio", "Lg",
    "Gionee", "Letv", "Redmi", "Ikall", "Lyf", "Oukitel", "Lenovo",
    "Zte", "Micromax", "Doogee", "Zanco", "Tesla", "Cat", "Tcl",
    "Vertu", "Sharp", "Royole", "Namotel", "Cola", "Xtouch",
    "Leeco", "Duoqin", "Blu"
]
st.title("📱 Smartphone Price Predictor")

st.subheader("Brand")

brand = st.selectbox("Brand", brands)
brand_df = pd.DataFrame({"brand" : [brand]})
brand_encoded = brand_encoder.transform(brand_df).iloc[0, 0]

st.subheader("Basic Specs")

rating = st.number_input("Rating (0-100)", min_value=0, max_value=100, step=1)
clock_speed = st.number_input("Clock Speed (GHz, 1.0-4.6)", min_value=1.0, max_value=4.6, step=0.01)
ram = st.number_input("RAM (GB)", min_value=1, max_value=24, step=1)
ram_tier = st.selectbox("RAM Tier", [1, 2, 3, 4])
fast_charging_power = st.number_input("Fast Charging Power (W, 0-250)",min_value=0, max_value=250, step=5)
memory_card_support = st.checkbox("Memory Card Support", value=True)
if memory_card_support:
    memory_card_size = st.number_input("Memory Card Size (GB)", min_value=0, max_value=2048, step=1) 
else:
    st.text_input("Memory Card Size (GB)", value = "", disabled=True, placeholder="No Memory Card Support")      

storage = st.number_input("Storage (GB)", min_value=4, max_value=2048, step=1)

st.subheader("Connectivity")
has_5g = st.checkbox("5G Support")
has_vo5g = st.checkbox("Vo5G")
has_nfc = st.checkbox("NFC")

st.subheader("Camera")
rear_cam = st.number_input("Primary Rear Camera (MP)", min_value=2, max_value=200, step=1)
front_cam = st.number_input("Primary Front Camera (MP)", min_value=1, max_value=100, step=1)
num_rear = st.number_input("Number of Rear Cameras", min_value=1, max_value=5, step=1)

st.subheader("Display")
screen_size = st.number_input("Screen Size (inches)", min_value=3.0, max_value=10.0, step=0.01)
res_width = st.number_input("Resolution Width", min_value=128, max_value=4000)
res_height = st.number_input("Resolution Height", min_value=128, max_value=4000)
refresh_rate = st.number_input("Refresh Rate (Hz)", min_value=30, max_value=240, step=1)

notch_type = st.selectbox(
    "Notch Type",
    [
        "Dual Punch Hole",
        "Large Notch",
        "No Notch",
        "Punch Hole",
        "Small Notch",
        "Water Drop Notch"
    ]
)

st.subheader("Operating System")

os_version = st.number_input("OS Version (1.0 - 20.0)", min_value=1.0, max_value=20.0, step=0.01)
os_name = st.selectbox("Operating System", ["Android", "iOS", "EMUI", "HarmonyOS", "KAI OS", "Pragati OS"])

processor_brand = st.selectbox( "Processor Brand", ["Snapdragon", "Bionic", "Dimensity", "Exynos", "Google Tensor", "Helio", "Kirin", "Unisoc", "Other"])
if st.button("Predict Price Category"):
    x = np.zeros(len(features))

    base_features = {
        'rating': rating,
        '5G': int(has_5g),
        'Vo5G': int(has_vo5g),
        'NFC': int(has_nfc),
        'Clock_Speed_GHz': clock_speed,
        'RAM Size GB': ram,
        'Storage Size GB': storage,
        'RAM Tier': ram_tier,
        'fast_charging_power': fast_charging_power,
        'Screen_Size': screen_size,
        'Resolution_Width': res_width,
        'Resolution_Height': res_height,
        'Refresh_Rate': refresh_rate,
        'primary_rear_camera_mp': rear_cam,
        'num_rear_cameras': num_rear,
        'primary_front_camera_mp': front_cam,
        'memory_card_support': int(memory_card_support),
        'os_version': os_version,
        'memory_card_size_gb': memory_card_size,
        'brand_encoded': brand_encoded
    }

    for i, feature in enumerate(features):
        if feature in base_features:
            x[i] = base_features[feature]
        elif feature == f"Notch_Type_{notch_type}":
            x[i] = 1
        elif feature == f"os_name_{os_name}":
            x[i] = 1
        elif feature == f"Processor_Brand_{processor_brand}":
            x[i] = 1
    prediction = model.predict([x])[0]

    if prediction == 1:
        st.markdown("""
        <div style="display:flex;justify-content:center;margin-top:40px;">
            <div style="
                background: linear-gradient(135deg,#00f5ff,#00c3ff);
                color:black;
                padding:35px 60px;
                border-radius:25px;
                font-size:2.2rem;
                font-weight:800;
                box-shadow:0 0 40px rgba(0,245,255,0.9);
            ">
                💰 Expensive Smartphone
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex;justify-content:center;margin-top:40px;">
            <div style="
                background: linear-gradient(135deg,#434343,#000);
                color:white;
                padding:35px 60px;
                border-radius:25px;
                font-size:2.2rem;
                font-weight:800;
                box-shadow:0 0 40px rgba(0,0,0,0.8);
            ">
                📉 Not Expensive Smartphone
            </div>
        </div>
        """, unsafe_allow_html=True)

