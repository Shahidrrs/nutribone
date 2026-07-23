import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from PIL import Image
import json
import os
import gdown

# ── Page config ──
st.set_page_config(
    page_title="NutriBone — Knee Osteoporosis Classifier",
    page_icon="🦴",
    layout="centered"
)

# ── Google Drive file IDs ──
DRIVE_IDS = {
    "vgg19_best.keras":           "1dq7XXW7oxogyTbZ91xpD8qTmgGZXi1lx",
    "resnet50_best.keras":        "1aDlLwZMATRx_w2YnrWvpXrX8SkB1koUW",
    "effnet_best.keras":          "1AxIHLjvffDgFwC3mtwN-rAiWRscmQxjm",
    "resnet50v2_best.keras":      "1PM2vmA08nBoP-a4kOASBlyQmwS5yXFjr",
    "best_ensemble_weights.json": "1aW_OJgjgIVU87lkGYMT4TqB8fZd9DE83",
}

CLASS_NAMES = ['Normal', 'Osteopenia', 'Osteoporosis']

CLASS_INFO = {
    'Normal': {
        'emoji': '🟢',
        'desc': 'Bone density is within the healthy range. No signs of bone loss detected.',
        'advice': 'Maintain a calcium-rich diet, regular weight-bearing exercise, and routine checkups.',
        'color': '#16a34a'
    },
    'Osteopenia': {
        'emoji': '🟠',
        'desc': 'Bone density is slightly lower than normal. This is an early warning stage before osteoporosis.',
        'advice': 'Consult a doctor. Lifestyle changes including diet and exercise can prevent progression.',
        'color': '#ea580c'
    },
    'Osteoporosis': {
        'emoji': '🔴',
        'desc': 'Significant bone density loss detected. Bones are weak and fracture risk is high.',
        'advice': 'Seek medical attention immediately. Treatment can help slow further bone loss.',
        'color': '#dc2626'
    }
}

PREPROCESS_FNS = {
    "vgg19_best.keras":      tf.keras.applications.vgg19.preprocess_input,
    "resnet50_best.keras":   tf.keras.applications.resnet50.preprocess_input,
    "effnet_best.keras":     tf.keras.applications.efficientnet.preprocess_input,
    "resnet50v2_best.keras": tf.keras.applications.resnet_v2.preprocess_input,
}

# ── Download models from Google Drive ──
def download_models():
    for fname, file_id in DRIVE_IDS.items():
        if not os.path.exists(fname):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, fname, quiet=False)

# ── Load models ──
@st.cache_resource(show_spinner=False)
def load_all():
    with st.spinner("Downloading models from Google Drive... (first time only, please wait)"):
        download_models()

    models = {}
    for fname in ["vgg19_best.keras", "resnet50_best.keras",
                  "effnet_best.keras", "resnet50v2_best.keras"]:
        models[fname] = tf.keras.models.load_model(fname)

    with open("best_ensemble_weights.json", "r") as f:
        data = json.load(f)
    weights = data["weights"]

    return models, weights

# ── TTA prediction ──
def predict_with_tta(model, preprocess_fn, img_array, n_tta=10):
    aug = ImageDataGenerator(
        rotation_range=5,
        horizontal_flip=True,
        zoom_range=0.03,
    )
    preds = np.zeros(3)
    img_expanded = np.expand_dims(img_array, axis=0)
    for _ in range(n_tta):
        for batch in aug.flow(img_expanded, batch_size=1):
            preprocessed = preprocess_fn(batch.astype(np.float32))
            pred = model.predict(preprocessed, verbose=0)
            preds += pred[0]
            break
    return preds / n_tta

# ── Main prediction ──
def run_prediction(image, models, weights, n_tta=10):
    img = image.convert('RGB').resize((224, 224))
    img_array = img_to_array(img)

    model_names = list(PREPROCESS_FNS.keys())
    all_preds = []

    for i, fname in enumerate(model_names):
        pred = predict_with_tta(
            models[fname],
            PREPROCESS_FNS[fname],
            img_array.copy(),
            n_tta=n_tta
        )
        all_preds.append((weights[i], pred))

    total_weight = sum(w for w, _ in all_preds)
    ensemble = np.zeros(3)
    for w, pred in all_preds:
        ensemble += (w / total_weight) * pred

    pred_idx = np.argmax(ensemble)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = ensemble[pred_idx] * 100

    return pred_class, confidence, ensemble

# ══════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════

st.markdown("""
<h1 style='text-align:center; color:#0d7377;'>🦴 NutriBone</h1>
<p style='text-align:center; color:#64748b; font-size:18px;'>
Deep Learning-Based Knee Osteoporosis Classification
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# About section
with st.expander("ℹ️ About this app"):
    st.markdown("""
    **NutriBone** classifies knee X-ray images into 3 categories:
    - 🟢 **Normal** — Healthy bone density
    - 🟠 **Osteopenia** — Early bone density loss (pre-osteoporosis)
    - 🔴 **Osteoporosis** — Significant bone density loss

    **Model:** Weighted ensemble of 4 deep learning models:
    VGG-19, ResNet50, EfficientNetB0, ResNet50V2

    **Test Accuracy:** 89.46% (3-class) | 93.01% (binary external test)

    **Training Data:** Combined Kaggle knee osteoporosis datasets (~2,319 images)
    """)

# Disclaimer
st.warning("""
⚠️ **Medical Disclaimer:** This tool is for **research and educational purposes only**.
It is NOT a substitute for professional medical diagnosis.
Always consult a qualified healthcare professional.
""")

st.markdown("---")

# Load models
try:
    models, weights = load_all()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Upload image
st.subheader("Upload a Knee X-Ray Image")
uploaded_file = st.file_uploader(
    "Choose an X-ray image (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)

    with col2:
        n_tta = st.slider("TTA Rounds (higher = more accurate but slower)", 5, 20, 10)

        if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
            with st.spinner("Analyzing... please wait"):
                try:
                    pred_class, confidence, ensemble = run_prediction(
                        image, models, weights, n_tta=n_tta)

                    info = CLASS_INFO[pred_class]

                    st.markdown(f"""
                    <div style='background-color:{info["color"]}22;
                                border-left: 5px solid {info["color"]};
                                padding: 15px; border-radius: 8px;
                                margin-top: 10px;'>
                        <h2 style='color:{info["color"]}; margin:0;'>
                            {info["emoji"]} {pred_class}
                        </h2>
                        <p style='font-size:18px; margin:5px 0;'>
                            Confidence: <b>{confidence:.1f}%</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")

    # Show results below if prediction was made
    if uploaded_file and st.session_state.get("predicted"):
        pass

    # Confidence chart
    if uploaded_file is not None:
        if st.button("🔍 Analyze X-Ray", key="analyze2"):
            pass

# Show full results after prediction
st.markdown("---")

if uploaded_file is not None:
    if st.button("Run Full Analysis", type="primary"):
        with st.spinner("Running full analysis with TTA..."):
            pred_class, confidence, ensemble = run_prediction(
                image, models, weights, n_tta=10)
            info = CLASS_INFO[pred_class]

            st.markdown("### Results")

            # Confidence bars for all 3 classes
            for i, cls in enumerate(CLASS_NAMES):
                pct = ensemble[i] * 100
                color = CLASS_INFO[cls]['color']
                st.markdown(f"**{CLASS_INFO[cls]['emoji']} {cls}**")
                st.progress(float(ensemble[i]))
                st.markdown(f"`{pct:.1f}%`")

            st.markdown("### What this means")
            st.info(info['desc'])

            st.markdown("### Recommended Action")
            st.success(info['advice'])

st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#94a3b8; font-size:12px;'>
NutriBone | Deep Learning Knee Osteoporosis Classification |
Research Project | Not for clinical use
</p>
""", unsafe_allow_html=True)
