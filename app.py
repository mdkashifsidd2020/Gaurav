import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import time
import os
from googletrans import Translator

# Model paths
MODEL_PATH = "rice_classifier.h5"
LABEL_PATH = "labels.txt"

# Load model
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    st.error("❌ Model or labels not found. Train using 'train_model.py'")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)
labels = open(LABEL_PATH).read().splitlines()
translator = Translator()
prediction_log = []

# UI setup
st.set_page_config(page_title="Rice Classifier", layout="centered")
st.title("🍚 Rice Type Classifier")

# Mode selection
mode = st.radio("Choose input mode:", ["Upload Image", "Use Webcam"])

def predict_image(image_np, image_name="webcam_image"):
    # Preprocess
    img_resized = np.array(image_np.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    # Predict
    start_time = time.time()
    prediction = model.predict(img_array)[0]
    end_time = time.time()

    top_indices = np.argsort(prediction)[::-1][:3]
    top_labels = [(labels[i], prediction[i] * 100) for i in top_indices]

    try:
        translated = translator.translate(top_labels[0][0], dest='hi').text
    except:
        translated = "⚠️ Translation failed"

    if top_labels[0][0].lower() == "not_rice" or top_labels[0][1] < 75:
        st.error("❌ This image doesn't appear to be a rice grain.")
    else:
        st.success(f"🌾 Type: {top_labels[0][0]} ({top_labels[0][1]:.2f}%)")
        st.markdown(f"🌐 In Hindi: **{translated}**")

    st.write(f"⏱ Prediction Time: `{end_time - start_time:.3f} sec`")

    # Chart
    prob_df = pd.DataFrame({
        "Rice Type": labels,
        "Confidence": prediction * 100
    })
    chart = alt.Chart(prob_df).mark_bar().encode(
        x=alt.X("Rice Type", sort="-y"),
        y="Confidence",
        color=alt.value("#4e79a7")
    ).properties(width=600, height=400)
    st.altair_chart(chart)

    # Top 3
    st.markdown("### 🔝 Top 3 Predictions")
    for label, score in top_labels:
        st.write(f"- {label}: **{score:.2f}%**")

    # Log
    prediction_log.append({
        "Image": image_name,
        "Top Prediction": top_labels[0][0],
        "Confidence (%)": round(top_labels[0][1], 2),
        "Prediction Time (s)": round(end_time - start_time, 3)
    })

if mode == "Upload Image":
    uploaded_files = st.file_uploader("Upload rice image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            st.divider()
            st.subheader(f"📷 {file.name}")
            image = Image.open(file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=300)
            predict_image(image, file.name)

elif mode == "Use Webcam":
    st.info("📸 Capture a rice image using your webcam")
    webcam_image = st.camera_input("Take a photo")
    if webcam_image:
        image = Image.open(webcam_image).convert("RGB")
        st.image(image, caption="Captured Image", width=300)
        predict_image(image, "webcam_capture")

# Download log
if prediction_log:
    df_log = pd.DataFrame(prediction_log)
    csv = df_log.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Prediction Log", data=csv, file_name="rice_predictions.csv", mime="text/csv")
