from roboflow import Roboflow
import os

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise RuntimeError("ROBOFLOW_API_KEY not set in this terminal session")

rf = Roboflow(api_key=api_key)
project = rf.workspace("driver-monitoring").project("dmd-tfiw0")
dataset = project.version(5).download("yolov8", location="../../data/dmd_full")

print("Downloaded to:", dataset.location)