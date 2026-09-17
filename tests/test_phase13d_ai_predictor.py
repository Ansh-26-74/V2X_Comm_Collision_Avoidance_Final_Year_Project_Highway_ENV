"""Phase 13D Step 2: Unit and Integration Tests for AI Trajectory Predictor.

Validates:
1. Model and scaler loading.
2. Five-sample history requirement and insufficient-history behavior.
3. Correct 40-feature construction and standardization.
4. Exactly six predicted future waypoints (+0.25s to +1.50s).
5. Accurate absolute waypoint coordinate conversion.
6. Deterministic inference across multiple calls.
7. Rejection of invalid, NaN, inf, or duplicate telemetry.
8. Independent history isolation per vehicle ID.
9. Structured safe fallback on model/prediction exceptions.
10. Decoupled geometric conflict evaluation operating independently of the neural network.
"""

import math
import os
import unittest
import numpy as np

from ai.intersection_ai.predictor import (
    IntersectionTrajectoryPredictor,
    AIPredictionResult,
    TrajectoryWaypoint,
    GeometricConflictAnalyzer,
    FORECAST_HORIZONS,
    HISTORY_LENGTH,
)
from v2i.v2v_inter_manager import IntersectionV2VMessage


class TestPhase13DAIPredictor(unittest.TestCase):
    """Test suite for Phase 13D Step 2: AI Inference Engine."""

    def setUp(self):
        """Instantiate predictor pointing to trained artifacts."""
        self.predictor = IntersectionTrajectoryPredictor(base_dir=".")

    def test_01_model_and_scaler_loading(self):
        """Test 1: Verify model and scaler load cleanly and are online."""
        self.assertTrue(self.predictor.is_ready)
        self.assertEqual(self.predictor.status, "ONLINE")
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.scaler)

        # Missing paths raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            IntersectionTrajectoryPredictor(base_dir=".", model_path="nonexistent.pth")

    def test_02_five_sample_history_requirement(self):
        """Test 2: Verifies inference requires exactly 5 valid telemetry packets."""
        self.assertEqual(self.predictor.get_history_count("C-01"), 0)

        for i in range(4):
            accepted = self.predictor.add_telemetry({
                "sender_id": "C-01",
                "x": 592.0,
                "y": 650.0 - i * 8.0,
                "vx": 0.0,
                "vy": -1.0,
                "speed": 80.0,
                "heading": -math.pi / 2.0,
                "hazard_status": "NORMAL",
                "timestamp": 1.0 + i * 0.1,
            })
            self.assertTrue(accepted)
            self.assertEqual(self.predictor.get_history_count("C-01"), i + 1)
            # Not ready yet
            res = self.predictor.predict("C-01")
            self.assertFalse(res.prediction_available)
            self.assertEqual(res.ai_risk_state, "INSUFFICIENT_DATA")

        # 5th packet fills buffer
        accepted = self.predictor.add_telemetry({
            "sender_id": "C-01",
            "x": 592.0,
            "y": 618.0,
            "vx": 0.0,
            "vy": -1.0,
            "speed": 80.0,
            "heading": -math.pi / 2.0,
            "hazard_status": "NORMAL",
            "timestamp": 1.4,
        })
        self.assertTrue(accepted)
        self.assertEqual(self.predictor.get_history_count("C-01"), 5)

        # Now prediction is available
        res = self.predictor.predict("C-01")
        self.assertTrue(res.prediction_available)
        self.assertEqual(len(res.waypoints), 6)

    def test_03_insufficient_history_behavior(self):
        """Test 3: Buffer with < 5 samples returns structured INSUFFICIENT_DATA result without error."""
        res = self.predictor.predict("UNKNOWN-VEHICLE")
        self.assertFalse(res.prediction_available)
        self.assertEqual(res.ai_risk_state, "INSUFFICIENT_DATA")
        self.assertEqual(len(res.waypoints), 0)
        self.assertIn("BUFFER_NOT_FULL", res.status_message)

    def test_04_correct_40_feature_construction(self):
        """Test 4: Verify 40-feature array has exact shape (1, 40) and normalized values."""
        for i in range(5):
            self.predictor.add_telemetry({
                "sender_id": "C-01",
                "x": 592.0,
                "y": 700.0 - i * 8.0,
                "vx": 0.0,
                "vy": -1.0,
                "speed": 80.0,
                "heading": -math.pi / 2.0,
                "hazard_status": "NORMAL",
                "timestamp": 10.0 + i * 0.1,
            })

        feats = self.predictor.build_features("C-01")
        self.assertIsNotNone(feats)
        self.assertEqual(feats.shape, (1, 40))

        # Check last step relative coordinates (dx, dy relative to latest sample are 0.0)
        curr_step_idx = (HISTORY_LENGTH - 1) * 8
        self.assertAlmostEqual(feats[0, curr_step_idx], 0.0, places=4)      # dx = 0
        self.assertAlmostEqual(feats[0, curr_step_idx + 1], 0.0, places=4)  # dy = 0

    def test_05_six_predicted_waypoints_and_horizons(self):
        """Test 5: Prediction produces exactly 6 waypoints at +0.25s, +0.50s, +0.75s, +1.00s, +1.25s, +1.50s."""
        for i in range(5):
            self.predictor.add_telemetry({
                "sender_id": "C-01",
                "x": 592.0,
                "y": 680.0 - i * 9.0,
                "vx": 0.0,
                "vy": -1.0,
                "speed": 90.0,
                "heading": -math.pi / 2.0,
                "hazard_status": "NORMAL",
                "timestamp": 2.0 + i * 0.1,
            })

        res = self.predictor.predict("C-01")
        self.assertTrue(res.prediction_available)
        self.assertEqual(len(res.waypoints), 6)
        self.assertEqual(res.horizon_seconds, 1.50)

        for i, wp in enumerate(res.waypoints):
            self.assertAlmostEqual(wp.horizon_offset_s, FORECAST_HORIZONS[i], places=3)
            self.assertIsInstance(wp.x, float)
            self.assertIsInstance(wp.y, float)

    def test_06_absolute_waypoint_conversion(self):
        """Test 6: Waypoint positions are converted from relative offsets to absolute world coordinates."""
        latest_y = 600.0
        for i in range(5):
            self.predictor.add_telemetry({
                "sender_id": "C-01",
                "x": 592.0,
                "y": latest_y + (4 - i) * 8.0,  # progressing from 632 down to 600
                "vx": 0.0,
                "vy": -1.0,
                "speed": 80.0,
                "heading": -math.pi / 2.0,
                "hazard_status": "NORMAL",
                "timestamp": 5.0 + i * 0.1,
            })

        res = self.predictor.predict("C-01")
        self.assertTrue(res.prediction_available)

        # Northbound vehicle: waypoints must continue moving North (y decreases from 600.0)
        prev_y = latest_y
        for wp in res.waypoints:
            self.assertLess(wp.y, prev_y, f"Waypoint y ({wp.y}) must be further North than prev_y ({prev_y})")
            prev_y = wp.y

        # Lane X should remain near 592.0
        for wp in res.waypoints:
            self.assertAlmostEqual(wp.x, 592.0, delta=5.0)

    def test_07_deterministic_repeated_inference(self):
        """Test 7: Repeated inference on unchanged history produces bitwise identical waypoints."""
        for i in range(5):
            self.predictor.add_telemetry({
                "sender_id": "C-01",
                "x": 592.0,
                "y": 640.0 - i * 7.5,
                "vx": 0.0,
                "vy": -1.0,
                "speed": 75.0,
                "heading": -math.pi / 2.0,
                "hazard_status": "NORMAL",
                "timestamp": 8.0 + i * 0.1,
            })

        res1 = self.predictor.predict("C-01")
        res2 = self.predictor.predict("C-01")

        self.assertTrue(res1.prediction_available)
        self.assertTrue(res2.prediction_available)

        for wp1, wp2 in zip(res1.waypoints, res2.waypoints):
            self.assertEqual(wp1.x, wp2.x)
            self.assertEqual(wp1.y, wp2.y)
            self.assertEqual(wp1.horizon_offset_s, wp2.horizon_offset_s)

    def test_08_invalid_and_nan_telemetry_handling(self):
        """Test 8: Rejects NaNs, infinities, out-of-order timestamps, and malformed telemetry."""
        # Valid sample
        self.assertTrue(self.predictor.add_telemetry({
            "sender_id": "C-01", "x": 592.0, "y": 700.0, "speed": 80.0, "timestamp": 1.0
        }))

        # NaN coordinate
        self.assertFalse(self.predictor.add_telemetry({
            "sender_id": "C-01", "x": float("nan"), "y": 690.0, "speed": 80.0, "timestamp": 1.1
        }))

        # Inf coordinate
        self.assertFalse(self.predictor.add_telemetry({
            "sender_id": "C-01", "x": 592.0, "y": float("inf"), "speed": 80.0, "timestamp": 1.1
        }))

        # Duplicate / retrograde timestamp (1.0 <= 1.0)
        self.assertFalse(self.predictor.add_telemetry({
            "sender_id": "C-01", "x": 592.0, "y": 690.0, "speed": 80.0, "timestamp": 1.0
        }))

        # Malformed / missing required field
        self.assertFalse(self.predictor.add_telemetry({"sender_id": "C-01"}))

        # None object
        self.assertFalse(self.predictor.add_telemetry(None))

        # Buffer only accepted the 1 valid packet
        self.assertEqual(self.predictor.get_history_count("C-01"), 1)

    def test_09_independent_history_for_multiple_vehicles(self):
        """Test 9: Multiple vehicles maintain completely isolated telemetry buffers."""
        for i in range(5):
            self.predictor.add_telemetry({
                "sender_id": "C-01",
                "x": 592.0,
                "y": 650.0 - i * 5.0,
                "speed": 50.0,
                "timestamp": 1.0 + i * 0.1,
            })

        # C-02 only has 2 samples
        for i in range(2):
            self.predictor.add_telemetry({
                "sender_id": "C-02",
                "x": 628.0,
                "y": 700.0 - i * 10.0,
                "speed": 100.0,
                "timestamp": 1.0 + i * 0.1,
            })

        self.assertEqual(self.predictor.get_history_count("C-01"), 5)
        self.assertEqual(self.predictor.get_history_count("C-02"), 2)

        res_c01 = self.predictor.predict("C-01")
        res_c02 = self.predictor.predict("C-02")

        self.assertTrue(res_c01.prediction_available)
        self.assertFalse(res_c02.prediction_available)

    def test_10_safe_fallback_on_model_exception(self):
        """Test 10: If model throws an exception, predictor returns structured safe fallback without crashing."""
        # Corrupt model reference intentionally
        original_model = self.predictor.model
        self.predictor.model = None

        # Add 5 samples
        for i in range(5):
            self.predictor.add_telemetry({
                "sender_id": "C-01", "x": 592.0, "y": 650.0 - i * 8.0, "speed": 80.0, "timestamp": 1.0 + i * 0.1
            })

        res = self.predictor.predict("C-01")
        self.assertFalse(res.prediction_available)
        self.assertEqual(res.ai_risk_state, "INSUFFICIENT_DATA")
        self.assertIn("INFERENCE_ERROR", res.status_message)

        # Restore model
        self.predictor.model = original_model

    def test_11_geometric_conflict_detection_decoupled(self):
        """Test 11: GeometricConflictAnalyzer operates independently on waypoints without neural network."""
        analyzer = GeometricConflictAnalyzer(corridor_half_width=24.0, safe_bumper_buffer=35.0)

        # Scenario A: Clear corridor (lead vehicle waypoints are in Lane 2 at x=628.0, ambulance in Lane 1 at x=592.0)
        lane2_waypoints = [
            TrajectoryWaypoint(horizon_offset_s=0.25, x=628.0, y=550.0),
            TrajectoryWaypoint(horizon_offset_s=0.50, x=628.0, y=530.0),
            TrajectoryWaypoint(horizon_offset_s=0.75, x=628.0, y=510.0),
            TrajectoryWaypoint(horizon_offset_s=1.00, x=628.0, y=490.0),
            TrajectoryWaypoint(horizon_offset_s=1.25, x=628.0, y=470.0),
            TrajectoryWaypoint(horizon_offset_s=1.50, x=628.0, y=450.0),
        ]

        conflict, min_dist, state = analyzer.evaluate(
            waypoints=lane2_waypoints,
            amb_x=592.0,
            amb_y=680.0,
            amb_speed=95.0,
        )
        self.assertFalse(conflict)
        self.assertEqual(state, "PREDICTED_SAFE")

        # Scenario B: Obstructed corridor (lead vehicle waypoints stopped ahead in Lane 1 at x=592.0, y=580.0)
        # Ambulance is at y=650.0, traveling at 95 px/s Northbound
        # In 1.0s, ambulance will be at 650 - 95 = 555 (penetrating lead at y=580)
        lane1_conflict_waypoints = [
            TrajectoryWaypoint(horizon_offset_s=0.25, x=592.0, y=585.0),
            TrajectoryWaypoint(horizon_offset_s=0.50, x=592.0, y=582.0),
            TrajectoryWaypoint(horizon_offset_s=0.75, x=592.0, y=580.0),
            TrajectoryWaypoint(horizon_offset_s=1.00, x=592.0, y=580.0),
            TrajectoryWaypoint(horizon_offset_s=1.25, x=592.0, y=580.0),
            TrajectoryWaypoint(horizon_offset_s=1.50, x=592.0, y=580.0),
        ]

        conflict, min_dist, state = analyzer.evaluate(
            waypoints=lane1_conflict_waypoints,
            amb_x=592.0,
            amb_y=650.0,
            amb_speed=95.0,
        )
        self.assertTrue(conflict)
        self.assertEqual(state, "PREDICTED_CONFLICT")
        self.assertLess(min_dist, 50.0)

    def test_12_v2v_message_compatibility(self):
        """Test 12: Predictor directly ingests standard IntersectionV2VMessage objects."""
        for i in range(5):
            cam = IntersectionV2VMessage(
                sender_id="C-01",
                receiver_id="AMB-01",
                vehicle_type="CIVILIAN",
                x=592.0,
                y=650.0 - i * 8.0,
                vx=0.0,
                vy=-1.0,
                speed=80.0,
                heading=-math.pi / 2.0,
                hazard_status="DECELERATING" if i >= 3 else "NORMAL",
                timestamp=12.0 + i * 0.1,
            )
            accepted = self.predictor.add_telemetry(cam)
            self.assertTrue(accepted)

        res = self.predictor.predict("C-01", ambulance_pos=(592.0, 720.0), ambulance_speed=95.0)
        self.assertTrue(res.prediction_available)
        self.assertEqual(res.sender_id, "C-01")
        self.assertGreater(res.inference_time_ms, 0.0)
        self.assertLess(res.inference_time_ms, 50.0)  # CPU inference well under 50ms

    def test_13_reset_functionality(self):
        """Test 13: Reset clears vehicle buffers cleanly."""
        self.predictor.add_telemetry({
            "sender_id": "C-01", "x": 592.0, "y": 650.0, "speed": 80.0, "timestamp": 1.0
        })
        self.assertEqual(self.predictor.get_history_count("C-01"), 1)

        self.predictor.reset("C-01")
        self.assertEqual(self.predictor.get_history_count("C-01"), 0)


if __name__ == "__main__":
    unittest.main()
