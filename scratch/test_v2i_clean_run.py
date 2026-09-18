import os
import sys
sys.path.insert(0, os.path.abspath("."))
os.environ["SDL_VIDEODRIVER"] = "dummy"

import time
import random
import pygame

import v2i_main

print("All imports succeeded. Testing v2i_main components...")

# Verify predictor loads instantly
t0 = time.perf_counter()
from ai.intersection_ai.predictor import IntersectionTrajectoryPredictor
pred = IntersectionTrajectoryPredictor(base_dir=".")
dt = (time.perf_counter() - t0) * 1000
print(f"IntersectionTrajectoryPredictor loaded in {dt:.2f} ms (is_ready={pred.is_ready}, status={pred.status})")
assert pred.is_ready, "Predictor should be ready"
assert dt < 500, f"Predictor initialization took too long: {dt:.2f} ms"

# Verify C-01 telemetry prediction
for i in range(5):
    accepted = pred.add_telemetry({
        "sender_id": "C-01",
        "x": 592.0,
        "y": 650.0 - i * 10.0,
        "vx": 0.0,
        "vy": -1.0,
        "speed": 80.0,
        "heading": -1.57,
        "hazard_status": "NORMAL",
        "timestamp": 1.0 + i * 0.1,
    })
    assert accepted, f"Telemetry sample {i} rejected"

res = pred.predict("C-01", ambulance_pos=(592.0, 700.0), ambulance_speed=95.0)
print(f"C-01 Prediction: available={res.prediction_available}, risk={res.ai_risk_state}, waypoints={len(res.waypoints)}, min_dist={res.predicted_min_distance:.1f}px")
assert res.prediction_available, "Prediction must be available after 5 samples"
assert len(res.waypoints) == 6, f"Expected 6 waypoints, got {len(res.waypoints)}"

print("Clean headless verification passed successfully!")
