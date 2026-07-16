import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(page_title="VisionFusion", layout="wide")
st.title("VisionFusion — Driver & Road Risk Dashboard")

API_URL = "http://127.0.0.1:8000/infer"

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
        with st.spinner("Running inference..."):
            files = {
                "driver_image": (driver_file.name, driver_file.getvalue(), driver_file.type),
                "road_image": (road_file.name, road_file.getvalue(), road_file.type),
            }
            response = requests.post(API_URL, files=files)

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

st.divider()
st.header("Road Hazard Detection — Potholes")

hazard_file = st.file_uploader("Road hazard image (pothole detection)", type=["jpg", "jpeg", "png"], key="hazard")
if hazard_file:
    st.image(hazard_file, caption="Road hazard feed", use_container_width=True)

    if st.button("Detect Potholes"):
        with st.spinner("Analyzing potholes..."):
            files = {"road_hazard_image": (hazard_file.name, hazard_file.getvalue(), hazard_file.type)}
            response = requests.post("http://127.0.0.1:8000/pothole_infer", files=files)

            if response.status_code == 200:
                result = response.json()
                detections = result["pothole_detections"]

                if not detections:
                    st.success("No potholes detected.")
                else:
                    st.subheader(f"{len(detections)} pothole(s) detected")
                    for i, det in enumerate(detections):
                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"Pothole {i+1} Confidence", f"{det['confidence']:.2f}")
                        col2.metric("Severity Score", f"{det['severity_score']:.3f}")
                        label = det["severity_label"]
                        if label == "Minor":
                            col3.success(label)
                        elif label == "Moderate":
                            col3.warning(label)
                        else:
                            col3.error(label)

                    st.caption("Note: severity is based on relative monocular depth (MiDaS), not a calibrated real-world measurement.")
            else:
                st.error(f"API error: {response.status_code} — {response.text}")

st.divider()
st.header("Full AI Safety Report")
st.caption("Runs all three models together and generates a combined safety analysis.")

col_a, col_b, col_c = st.columns(3)
with col_a:
    fr_driver = st.file_uploader("Driver image", type=["jpg", "jpeg", "png"], key="fr_driver")
with col_b:
    fr_road = st.file_uploader("Road image", type=["jpg", "jpeg", "png"], key="fr_road")
with col_c:
    fr_hazard = st.file_uploader("Road hazard image", type=["jpg", "jpeg", "png"], key="fr_hazard")

if fr_driver and fr_road and fr_hazard:
    if st.button("Generate Full Safety Report"):
        with st.spinner("Running full analysis..."):
            files = {
                "driver_image": (fr_driver.name, fr_driver.getvalue(), fr_driver.type),
                "road_image": (fr_road.name, fr_road.getvalue(), fr_road.type),
                "road_hazard_image": (fr_hazard.name, fr_hazard.getvalue(), fr_hazard.type),
            }
            response = requests.post("http://127.0.0.1:8000/full_report", files=files)

            if response.status_code == 200:
                data = response.json()
                report = data["report"]

                st.subheader("VisionFusion AI Safety Report")
                m1, m2, m3 = st.columns(3)
                m1.metric("Driver Safety Index", f"{report['driver_safety_index']}/100")
                m2.metric("Road Health Index", f"{report['road_health_index']}/100")
                risk = report["overall_risk"]
                if risk == "Low":
                    m3.success(f"Overall Risk: {risk}")
                elif risk == "Medium":
                    m3.warning(f"Overall Risk: {risk}")
                else:
                    m3.error(f"Overall Risk: {risk}")

                st.subheader("Analysis Summary")
                st.write(report["summary"])

                st.subheader("Recommendations")
                for rec in report["recommendations"]:
                    st.write(f"- {rec}")

                st.subheader("Annotated Feeds")
                img_col1, img_col2, img_col3 = st.columns(3)
                img_col1.image(base64.b64decode(data["driver_image_annotated"]), caption="Driver (annotated)")
                img_col2.image(base64.b64decode(data["road_image_annotated"]), caption="Road (annotated)")
                img_col3.image(base64.b64decode(data["hazard_image_annotated"]), caption="Hazard (annotated)")
            else:
                st.error(f"API error: {response.status_code} — {response.text}")

st.divider()
st.header("Road Hazard Detection — Video")
st.caption("Upload a driving video to detect potholes across multiple frames over time.")

hazard_video_file = st.file_uploader("Road hazard video", type=["mp4", "mov", "avi"], key="hazard_video")

if hazard_video_file:
    st.video(hazard_video_file)

    if st.button("Analyze Video for Potholes"):
        with st.spinner("Analyzing video — this may take a moment..."):
            files = {"road_hazard_video": (hazard_video_file.name, hazard_video_file.getvalue(), hazard_video_file.type)}
            response = requests.post("http://127.0.0.1:8000/video_pothole_infer", files=files)

            if response.status_code == 200:
                result = response.json()

                st.subheader("Video Analysis Summary")
                v1, v2, v3 = st.columns(3)
                v1.metric("Frames Analyzed", result["frames_analyzed"])
                v2.metric("Total Detections", result["total_detections"])
                worst = result["worst_severity"]
                if worst == "Severe":
                    v3.error(f"Worst Severity: {worst}")
                elif worst == "Moderate":
                    v3.warning(f"Worst Severity: {worst}")
                elif worst == "Minor":
                    v3.success(f"Worst Severity: {worst}")
                else:
                    v3.info("No potholes detected")

                st.subheader("Detections Over Time")
                for frame_result in result["frame_results"]:
                    ts = frame_result["timestamp_sec"]
                    dets = frame_result["detections"]
                    st.write(f"**t = {ts}s** — {len(dets)} pothole(s): " +
                             ", ".join(f"{d['severity_label']} ({d['confidence']:.2f})" for d in dets))

                st.caption("Note: frame-independent sampling — the same physical pothole may be counted more than once if it appears in multiple sampled frames.")
            else:
                st.error(f"API error: {response.status_code} — {response.text}")