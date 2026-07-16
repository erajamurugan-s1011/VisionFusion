import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "fusion"))

from fusion_model import driver_risk_score, road_risk_score, combine_risk
from safety_report import (
    compute_driver_safety_index,
    compute_road_health_index,
    compute_overall_risk,
    generate_recommendations,
)


def test_driver_risk_score_empty():
    assert driver_risk_score([]) == 0.0


def test_driver_risk_score_safe_driving():
    score = driver_risk_score([("SafeDriving", 0.95)])
    assert score == 0.0


def test_driver_risk_score_dangerous():
    score = driver_risk_score([("DangerousDriving", 1.0)])
    assert score == 1.0


def test_road_risk_score_empty():
    assert road_risk_score([], 1, 1) == 0.0


def test_road_risk_score_close_pedestrian_higher_than_far_car():
    close_person = [("person", 0.9, (0.5, 0.9, 0.3, 0.3))]
    far_car = [("car", 0.9, (0.5, 0.1, 0.05, 0.05))]
    person_score = road_risk_score(close_person, 1, 1)
    car_score = road_risk_score(far_car, 1, 1)
    assert person_score > car_score


def test_combine_risk_bounds():
    assert 0.0 <= combine_risk(1.0, 1.0) <= 1.0
    assert combine_risk(0.0, 0.0) == 0.0


def test_driver_safety_index_no_detections():
    index, state = compute_driver_safety_index([])
    assert index == 100


def test_driver_safety_index_dangerous_driving():
    index, state = compute_driver_safety_index([("DangerousDriving", 1.0)])
    assert index < 100
    assert state == "DangerousDriving"


def test_road_health_index_no_detections():
    index = compute_road_health_index([], [])
    assert index == 100


def test_road_health_index_with_severe_pothole():
    index = compute_road_health_index([], [{"severity_label": "Severe"}])
    assert index < 100


def test_overall_risk_thresholds():
    assert compute_overall_risk(90, 90) == "Low"
    assert compute_overall_risk(60, 60) == "Medium"
    assert compute_overall_risk(30, 30) == "High"


def test_recommendations_not_empty():
    recs = generate_recommendations("SafeDriving", 100, 100, [])
    assert len(recs) > 0


def test_recommendations_flag_sleepy_driving():
    recs = generate_recommendations("SleepyDriving", 50, 100, [])
    assert any("break" in r.lower() for r in recs)