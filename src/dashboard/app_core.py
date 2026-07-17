import streamlit as st
import requests

st.set_page_config(page_title="VisionFusion", layout="wide")
st.title("VisionFusion — Driver & Road Risk Dashboard")
st.caption("Live core demo: driver-state + road-state + fusion. Full pipeline (pothole + depth + video) available in the GitHub repo, runs locally.")

API_URL = "https://visionfusion-core.onrender.com/infer"

col1, col2 = st.columns(2)

with col1:
    driver_file = st.file_uploader("Driver-facing camera image", type=["jpg", "jpeg", "png"], key="driver")
    if driver_file:
        st.image(driver_file, caption="Driver feed", use_container_width=True)

with col2:
    road_file = st.file_uploader("Road-facing camera image", type=["jpg", "jpeg", "png"], key="road")
    if road_file:
        st.image(road_file, caption="Road feed", use_container_width=True)

if driver_file and road_file:
    if st.button("Run Fusion Inference"):
        with st.spinner("Running inference — free tier CPU is slow, this can take 30-90 seconds..."):
            files = {
                "driver_image": (driver_file.name, driver_file.getvalue(), driver_file.type),
                "road_image": (road_file.name, road_file.getvalue(), road_file.type),
            }
            try:
                response = requests.post(API_URL, files=files, timeout=120)

                if response.status_code == 200:
                    result = response.json()

                    st.subheader("Results")
                    risk_col1, risk_col2, risk_col3 = st.columns(3)
                    risk_col1.metric("Driver Risk", f"{result['driver_risk']:.3f}")
                    risk_col2.metric("Road Risk", f"{result['road_risk']:.3f}")
                    risk_col3.metric("Fused Risk", f"{result['fused_risk']:.3f}")

                    fused = result["fused_risk"]
                    if fused < 0.3:
                        st.success("Low risk")
                    elif fused < 0.6:
                        st.warning("Moderate risk")
                    else:
                        st.error("High risk")

                    st.subheader("Driver Detections")
                    st.write(result["driver_detections"])

                    st.subheader("Road Detections")
                    st.write(result["road_detections"])
                else:
                    st.error(f"API error: {response.status_code} — {response.text}")
            except requests.exceptions.Timeout:
                st.error("Request timed out — the free-tier server may still be waking up. Try again in a moment.")