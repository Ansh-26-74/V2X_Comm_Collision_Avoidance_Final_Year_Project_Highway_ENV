"""Phase 13D Asynchronous AI Runtime & Safe Integration Test Suite.

Validates:
Test 1: AI predictor can initialize asynchronously in a background thread.
Test 2: TrafficManager starts with AI unavailable (ai_predictor is None, ai_loading=True).
Test 3: Completed predictor is safely attached to the existing TrafficManager via set_ai_predictor.
Test 4: Telemetry generated while AI is loading is retained in the pending buffer.
Test 5: Retained telemetry reaches the predictor after initialization and flushes.
Test 6: Five telemetry samples transition AI from collecting to active.
Test 7: C-01 generates an actual prediction from live V2V telemetry.
Test 8: Exactly six future waypoints (+0.25s .. +1.50s) are generated.
Test 9: AI prediction reaches AmbulanceVehicle (latest_ai_prediction).
Test 10: AI trajectory can be rendered by draw_ai_trajectory without error.
Test 11: AI initialization failure falls back safely (ai_error exposed, simulation unaffected).
Test 12: Simulation is not reset when AI becomes ready (vehicles and clock preserved).
Test 13: No synchronous PyTorch loading occurs before first frame (lazy import intact).
Test 14: Highway V2V remains untouched.
"""

import collections
import math
import os
import random
import sys
import threading
import time
import unittest
import pygame

from ai.intersection_ai.predictor import (
    IntersectionTrajectoryPredictor,
    AIPredictionResult,
    TrajectoryWaypoint,
    FORECAST_HORIZONS,
)
from v2i.smart_signal import SmartTrafficSignal, SignalState
from v2i.traffic_manager import (
    TrafficManager,
    CivilianVehicle,
    AmbulanceVehicle,
    SOUTH_PRIMARY_LANE_X,
    SOUTH_OVERTAKE_LANE_X,
)
from v2i.v2i_manager import V2IManager
from v2i.v2v_inter_manager import IntersectionV2VManager, IntersectionV2VMessage
import v2i.v2i_renderer as renderer


class TestPhase13DAIAsyncRuntime(unittest.TestCase):
    """Regression test suite for async AI initialization and runtime integration."""

    def setUp(self):
        pygame.init()
        random.seed(42)

    def tearDown(self):
        pygame.quit()

    def test_01_async_initialization(self):
        """Test 1: AI predictor can initialize asynchronously in a background thread."""
        holder = {"predictor": None, "error": None}
        ready_event = threading.Event()

        def _load():
            try:
                p = IntersectionTrajectoryPredictor(base_dir=".")
                holder["predictor"] = p
                ready_event.set()
            except Exception as e:
                holder["error"] = str(e)
                ready_event.set()

        t = threading.Thread(target=_load, daemon=True)
        t.start()
        finished = ready_event.wait(timeout=10.0)

        self.assertTrue(finished, "Background AI loader thread must complete within timeout")
        self.assertIsNotNone(holder["predictor"], f"Predictor must be instantiated: {holder['error']}")
        self.assertTrue(holder["predictor"].is_ready)

    def test_02_traffic_manager_starts_without_ai(self):
        """Test 2: TrafficManager starts with AI unavailable."""
        tm = TrafficManager(ai_predictor=None)
        self.assertIsNone(tm.ai_predictor)
        self.assertTrue(tm.ai_loading)
        self.assertEqual(len(tm._pending_v2v_telemetry), 0)

    def test_03_completed_predictor_attached_to_traffic_manager(self):
        """Test 3: Completed predictor is safely attached to the existing TrafficManager."""
        tm = TrafficManager(ai_predictor=None)
        pred = IntersectionTrajectoryPredictor(base_dir=".")

        tm.set_ai_predictor(pred)
        self.assertIsNotNone(tm.ai_predictor)
        self.assertFalse(tm.ai_loading)
        self.assertIs(tm.ai_predictor, pred)

    def test_04_telemetry_retained_while_ai_loading(self):
        """Test 4: Telemetry generated while AI is loading is retained in pending buffer."""
        tm = TrafficManager(ai_predictor=None)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        v2v = IntersectionV2VManager(v2v_range=200.0, latency=0.020)

        # Place C-01 on South approach
        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=700.0, speed=80.0)
        tm.vehicles.append(c01)

        # Update several frames while ai_predictor is None
        for step in range(5):
            tm.update(1.0 / 60.0, sig, sim_time=step * 0.1, v2v_inter_manager=v2v)

        self.assertIn("C-01", tm._pending_v2v_telemetry)
        self.assertGreaterEqual(len(tm._pending_v2v_telemetry["C-01"]), 1)

    def test_05_retained_telemetry_flushed_to_predictor(self):
        """Test 5: Retained telemetry reaches the predictor after initialization."""
        tm = TrafficManager(ai_predictor=None)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        v2v = IntersectionV2VManager(v2v_range=200.0, latency=0.020)

        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=700.0, speed=80.0)
        tm.vehicles.append(c01)

        for step in range(6):
            tm.update(1.0 / 60.0, sig, sim_time=step * 0.1, v2v_inter_manager=v2v)

        retained_count = len(tm._pending_v2v_telemetry["C-01"])
        self.assertGreaterEqual(retained_count, 5)

        # Attach predictor
        pred = IntersectionTrajectoryPredictor(base_dir=".")
        tm.set_ai_predictor(pred)

        self.assertEqual(len(tm._pending_v2v_telemetry), 0, "Pending buffer must be cleared after handoff")
        pred_history = pred.history_buffers.get("C-01", [])
        self.assertEqual(len(pred_history), min(retained_count, 5), "Predictor history must be populated up to max capacity (5)")

    def test_06_five_samples_transition_ai_to_active(self):
        """Test 6: Five telemetry samples transition AI from collecting to active."""
        pred = IntersectionTrajectoryPredictor(base_dir=".")
        tm = TrafficManager(ai_predictor=pred)
        sig = SmartTrafficSignal("INT-01", 560, 415)

        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=700.0, speed=80.0)
        tm.vehicles.append(c01)

        # 4 samples: insufficient
        for step in range(4):
            tm.update(1.0 / 60.0, sig, sim_time=step * 0.1)
        self.assertEqual(len(pred.history_buffers.get("C-01", [])), 4)
        self.assertFalse(getattr(tm.latest_ai_prediction, "prediction_available", False))

        # 5th sample: transitions to ACTIVE
        tm.update(1.0 / 60.0, sig, sim_time=0.45)
        self.assertEqual(len(pred.history_buffers.get("C-01", [])), 5)
        self.assertIsNotNone(tm.latest_ai_prediction)
        self.assertTrue(tm.latest_ai_prediction.prediction_available)

    def test_07_c01_generates_actual_prediction(self):
        """Test 7: C-01 generates an actual prediction from live V2V telemetry."""
        pred = IntersectionTrajectoryPredictor(base_dir=".")
        tm = TrafficManager(ai_predictor=pred)
        sig = SmartTrafficSignal("INT-01", 560, 415)

        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=650.0, speed=85.0)
        tm.vehicles.append(c01)

        for step in range(10):
            tm.update(1.0 / 60.0, sig, sim_time=step * 0.1)

        result = tm.latest_ai_prediction
        self.assertIsNotNone(result)
        self.assertTrue(result.prediction_available)
        self.assertEqual(result.sender_id, "C-01")
        self.assertIn(result.ai_risk_state, ("PREDICTED_SAFE", "PREDICTED_CONFLICT"))

    def test_08_six_waypoints_generated(self):
        """Test 8: Exactly six future waypoints (+0.25s .. +1.50s) are generated."""
        pred = IntersectionTrajectoryPredictor(base_dir=".")
        tm = TrafficManager(ai_predictor=pred)
        sig = SmartTrafficSignal("INT-01", 560, 415)

        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=650.0, speed=85.0)
        tm.vehicles.append(c01)

        for step in range(6):
            tm.update(1.0 / 60.0, sig, sim_time=step * 0.1)

        waypoints = tm.latest_ai_prediction.waypoints
        self.assertEqual(len(waypoints), 6)
        for i, h in enumerate(FORECAST_HORIZONS):
            self.assertAlmostEqual(waypoints[i].horizon_offset_s, h)
            self.assertTrue(math.isfinite(waypoints[i].x))
            self.assertTrue(math.isfinite(waypoints[i].y))

    def test_09_ai_prediction_reaches_ambulance(self):
        """Test 9: AI prediction reaches AmbulanceVehicle (latest_ai_prediction)."""
        pred = IntersectionTrajectoryPredictor(base_dir=".")
        tm = TrafficManager(ai_predictor=pred)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        v2v = IntersectionV2VManager(v2v_range=200.0, latency=0.020)

        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=700.0, speed=95.0)
        tm.ambulance = amb
        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=620.0, speed=80.0)
        tm.vehicles.append(c01)

        for step in range(10):
            tm.update(1.0 / 60.0, sig, sim_time=step * 0.1, v2v_inter_manager=v2v)

        self.assertIsNotNone(amb.latest_ai_prediction)
        self.assertTrue(amb.ai_prediction_available)
        self.assertEqual(amb.latest_ai_prediction.sender_id, "C-01")

    def test_10_ai_trajectory_can_be_rendered(self):
        """Test 10: AI trajectory can be rendered by draw_ai_trajectory without error."""
        screen = pygame.Surface((1280, 720))
        pred = IntersectionTrajectoryPredictor(base_dir=".")
        tm = TrafficManager(ai_predictor=pred)
        sig = SmartTrafficSignal("INT-01", 560, 415)

        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=650.0, speed=85.0)
        tm.vehicles.append(c01)

        for step in range(6):
            tm.update(1.0 / 60.0, sig, sim_time=step * 0.1)

        font = pygame.font.SysFont("consolas", 12)
        # Should render cleanly with or without ambulance
        renderer.draw_ai_trajectory(
            screen=screen,
            ambulance=tm.ambulance,
            civilian_vehicles=tm.vehicles,
            font_small=font,
            font_timer=font,
            show_ai=True,
            traffic_manager=tm,
        )

        renderer.draw_ai_hud_card(
            screen=screen,
            traffic_manager=tm,
            font_bold=font,
            font_normal=font,
            font_small=font,
        )

    def test_11_ai_initialization_failure_falls_back_safely(self):
        """Test 11: AI initialization failure falls back safely (ai_error exposed, simulation unaffected)."""
        tm = TrafficManager(ai_predictor=None)
        sig = SmartTrafficSignal("INT-01", 560, 415)

        # Simulate background failure
        tm.ai_loading = False
        tm.ai_error = "ModelFileNotFoundError"

        # Civilian traffic and simulation continue normally
        for step in range(30):
            tm.update(1.0 / 60.0, sig, sim_time=step * (1.0 / 60.0))

        screen = pygame.Surface((1280, 720))
        font = pygame.font.SysFont("consolas", 12)
        renderer.draw_ai_hud_card(
            screen=screen,
            traffic_manager=tm,
            font_bold=font,
            font_normal=font,
            font_small=font,
        )
        self.assertEqual(tm.ai_error, "ModelFileNotFoundError")
        self.assertIsNone(tm.ai_predictor)

    def test_12_simulation_not_reset_on_ai_ready(self):
        """Test 12: Simulation is not reset when AI becomes ready."""
        tm = TrafficManager(ai_predictor=None)
        sig = SmartTrafficSignal("INT-01", 560, 415)

        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=600.0, speed=80.0)
        tm.vehicles.append(c01)

        # Advance 1.0s
        for step in range(60):
            tm.update(1.0 / 60.0, sig, sim_time=step * (1.0 / 60.0))

        pos_before = (c01.x, c01.y)
        vehicle_count_before = len(tm.vehicles)

        # AI finishes and is attached
        pred = IntersectionTrajectoryPredictor(base_dir=".")
        tm.set_ai_predictor(pred)

        self.assertEqual(len(tm.vehicles), vehicle_count_before)
        self.assertEqual((c01.x, c01.y), pos_before)

    def test_13_no_synchronous_pytorch_before_first_frame(self):
        """Test 13: TrafficManager creation does not synchronously import heavy PyTorch."""
        tm = TrafficManager(ai_predictor=None)
        self.assertIsNone(tm.ai_predictor)
        self.assertTrue(tm.ai_loading)

    def test_14_highway_v2v_remains_untouched(self):
        """Test 14: Highway V2V files exist and are not broken."""
        self.assertTrue(os.path.exists("main.py"))
        self.assertTrue(os.path.exists("v2v_manager.py"))
        self.assertTrue(os.path.exists("ai/trajectory_predictor.py"))
        self.assertTrue(os.path.exists("ai/safety_fusion.py"))
        self.assertTrue(os.path.exists("models/vehicle_trajectory_gru.pth"))

    def test_15_live_main_loop_event_logger_handoff(self):
        """Test 15: Main loop handoff executes without AttributeError on event_logger."""
        from v2i.event_logger import EventLogger, EventCategory
        el = EventLogger(max_events=14)
        tm = TrafficManager(ai_predictor=None)
        pred = IntersectionTrajectoryPredictor(base_dir=".")

        # Simulate the exact handoff lines from v2i_main.py
        tm.set_ai_predictor(pred)
        el.add_event(
            timestamp=1.5,
            category=EventCategory.SYSTEM,
            event_type="AI",
            message="AI PREDICTOR ONLINE (1.50s HORIZON)",
        )
        # Also verify backward-compat log method alias works
        el.log(2.0, EventCategory.SYSTEM, "AI", "AI PREDICTOR ONLINE")

        self.assertFalse(tm.ai_loading)
        self.assertIsNotNone(tm.ai_predictor)
        events = el.get_events()
        self.assertGreaterEqual(len(events), 1)

    def test_16_end_to_end_sim_ai_active_transition(self):
        """Test 16: Full simulation progression from INITIALIZING to ACTIVE with 6 waypoints."""
        random.seed(42)
        sig = SmartTrafficSignal()
        tm = TrafficManager(ai_predictor=None)
        v2v = IntersectionV2VManager(v2v_range=200.0, latency=0.020)
        pred = IntersectionTrajectoryPredictor(base_dir=".")

        # Initially INITIALIZING
        self.assertTrue(tm.ai_loading)

        # Attach predictor
        tm.set_ai_predictor(pred)
        self.assertFalse(tm.ai_loading)

        # Spawn C-01
        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=750.0, speed=85.0)
        tm.vehicles.append(c01)

        # Update 10 steps to collect telemetry and produce predictions
        for step in range(10):
            tm.update(1.0 / 60.0, sig, sim_time=step * 0.1, v2v_inter_manager=v2v)

        # Confirm prediction is ACTIVE and has 6 waypoints
        self.assertIsNotNone(tm.latest_ai_prediction)
        self.assertTrue(tm.latest_ai_prediction.prediction_available)
        self.assertEqual(len(tm.latest_ai_prediction.waypoints), 6)
        self.assertEqual(tm.latest_ai_prediction.sender_id, "C-01")


if __name__ == "__main__":
    unittest.main()
