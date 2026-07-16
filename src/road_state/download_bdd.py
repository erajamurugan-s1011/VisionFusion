from roboflow import Roboflow
import os

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise RuntimeError("ROBOFLOW_API_KEY not set in this terminal session")

rf = Roboflow(api_key=api_key)
project = rf.workspace("layout-3rcq8").project("bdd100k-axxue")
dataset = project.version(1).download("yolov8", location="../../data/bdd100k_full")

print("Downloaded to:", dataset.location)