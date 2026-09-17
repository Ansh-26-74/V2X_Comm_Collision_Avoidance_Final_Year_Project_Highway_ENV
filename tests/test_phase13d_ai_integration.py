"""Phase 13D Step 3: Live V2V + AI Trajectory Predictor Integration Tests.

Validates:
1. Predictor receives live V2V telemetry through the traffic manager update loop.
2. Accumulation of 5 valid telemetry packets enables AI inference.
3. C-01 gets a valid prediction during the integrated scenario.
4. Prediction contains exactly 6 waypoints at a 1.5-second horizon.
5. AI prediction does NOT alter existing TTC computation or risk state.
6. AI failure falls back safely without disrupting deterministic V2V logic.
7. Multiple vehicles maintain independent AI histories.
8. Duplicate/out-of-order telemetry packets are not double-counted.
9. Existing V2V evasive maneuver and clearance behavior remains completely intact.
10. Event logger captures AI lifecycle transitions cleanly.
"""

import math
import random
import unittest

from ai.intersection_ai.predictor import (
    IntersectionTrajectoryPredictor,
    AIPredictionResult,
    TrajectoryWaypoint,
    FORECAST_HORIZONS,
)
from v2i.event_logger import EventLogger, EventCategory
from v2i.smart_signal import SmartTrafficSignal
from v2i.traffic_manager import (
    TrafficManager,
    AmbulanceVehicle,
    CivilianVehicle,
    SOUTH_PRIMARY_LANE_X,
    SOUTH_OVERTAKE_LANE_X,
)
from v2i.v2i_manager import V2IManager
from v2i.v2v_inter_manager import IntersectionV2VManager, IntersectionV2VMessage


class TestPhase13DAIIntegration(unittest.TestCase):
    """Test suite for Phase 13D Step 3: Live V2V + AI Integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.predictor = IntersectionTrajectoryPredictor(base_dir=".")
        self.v2v = IntersectionV2VManager(v2v_range=200.0, latency=0.020)
        self.logger = EventLogger(max_events=20)

    def test_01_predictor_receives_live_v2v_telemetry(self):
        """Test 1: TrafficManager update loop feeds delivered V2V packets into the predictor."""
        tm = TrafficManager(enable_v2v_hazard=False, ai_predictor=self.predictor)
        sig = SmartTrafficSignal("INT-TEST", 560, 415)

        # Place ambulance and C-01 on South approach
        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=700.0, speed=95.0)
        tm.ambulance = amb
        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=600.0, speed=80.0)
        tm.vehicles.append(c01)

        # Run several frames at 10 Hz packet intervals
        sim_time = 0.0
        for step in range(30):
            sim_time += 1.0 / 60.0
            tm.update(
                dt=1.0 / 60.0,
                signal_controller=sig,
                sim_time=sim_time,
                v2v_inter_manager=self.v2v,
            )

        # Verify C-01 telemetry was received and buffered in the predictor
        history_count = self.predictor.get_history_count("C-01")
        self.assertGreaterEqual(history_count, 1, "Predictor must ingest delivered V2V telemetry")

    def test_02_five_samples_enable_inference_for_c01(self):
        """Test 2: After 5 V2V packets arrive, AI inference becomes available for C-01."""
        tm = TrafficManager(enable_v2v_hazard=False, ai_predictor=self.predictor)
        sig = SmartTrafficSignal("INT-TEST", 560, 415)

        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=720.0, speed=95.0)
        tm.ambulance = amb
        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=620.0, speed=80.0)
        tm.vehicles.append(c01)

        # 5 packets at 10 Hz require ~0.5s of simulation time (30-40 frames)
        sim_time = 0.0
        for step in range(45):
            sim_time += 1.0 / 60.0
            tm.update(
                dt=1.0 / 60.0,
                signal_controller=sig,
                sim_time=sim_time,
                v2v_inter_manager=self.v2v,
            )

        self.assertGreaterEqual(self.predictor.get_history_count("C-01"), 5)
        self.assertIsNotNone(amb.latest_ai_prediction)
        self.assertTrue(amb.latest_ai_prediction.prediction_available)
        self.assertTrue(amb.ai_prediction_available)
        self.assertEqual(len(amb.latest_ai_prediction.waypoints), 6)

    def test_03_prediction_contains_six_waypoints_and_1_5s_horizon(self):
        """Test 3: Valid AI prediction contains 6 waypoints spanning up to 1.50 seconds."""
        tm = TrafficManager(enable_v2v_hazard=False, ai_predictor=self.predictor)
        sig = SmartTrafficSignal("INT-TEST", 560, 415)

        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=700.0, speed=95.0)
        tm.ambulance = amb
        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=600.0, speed=80.0)
        tm.vehicles.append(c01)

        sim_time = 0.0
        for _ in range(50):
            sim_time += 1.0 / 60.0
            tm.update(
                dt=1.0 / 60.0,
                signal_controller=sig,
                sim_time=sim_time,
                v2v_inter_manager=self.v2v,
            )

        pred = amb.latest_ai_prediction
        self.assertIsNotNone(pred)
        self.assertTrue(pred.prediction_available)
        self.assertEqual(pred.horizon_seconds, 1.50)
        self.assertEqual(len(pred.waypoints), 6)

        for i, wp in enumerate(pred.waypoints):
            self.assertAlmostEqual(wp.horizon_offset_s, FORECAST_HORIZONS[i], places=3)
            self.assertIsInstance(wp.x, float)
            self.assertIsInstance(wp.y, float)
            # Waypoint Y coordinates must be in front of C-01 (progressing North)
            self.assertLess(wp.y, 650.0)

    def test_04_ai_prediction_does_not_alter_ttc_state(self):
        """Test 4: AI prediction is strictly advisory in Step 3; kinematic TTC is unchanged."""
        # Run identical scenario with and without AI predictor
        predictor_instance = IntersectionTrajectoryPredictor(base_dir=".")

        tm_with_ai = TrafficManager(enable_v2v_hazard=False, ai_predictor=predictor_instance)
        tm_no_ai   = TrafficManager(enable_v2v_hazard=False, ai_predictor=None)
        sig = SmartTrafficSignal("INT-TEST", 560, 415)
        v2v_1 = IntersectionV2VManager(v2v_range=200.0, latency=0.020)
        v2v_2 = IntersectionV2VManager(v2v_range=200.0, latency=0.020)

        amb_ai = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=700.0, speed=95.0)
        tm_with_ai.ambulance = amb_ai
        c01_ai = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=580.0, speed=80.0)
        tm_with_ai.vehicles.append(c01_ai)

        amb_no = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=700.0, speed=95.0)
        tm_no_ai.ambulance = amb_no
        c01_no = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=580.0, speed=80.0)
        tm_no_ai.vehicles.append(c01_no)

        sim_time = 0.0
        for _ in range(50):
            sim_time += 1.0 / 60.0
            tm_with_ai.update(1.0 / 60.0, sig, sim_time=sim_time, v2v_inter_manager=v2v_1)
            tm_no_ai.update(1.0 / 60.0, sig, sim_time=sim_time, v2v_inter_manager=v2v_2)

            # TTC and risk state must remain identical
            self.assertEqual(amb_ai.v2v_risk_state, amb_no.v2v_risk_state)
            if not math.isinf(amb_ai.v2v_ttc) and not math.isinf(amb_no.v2v_ttc):
                self.assertAlmostEqual(amb_ai.v2v_ttc, amb_no.v2v_ttc, places=3)

    def test_05_ai_failure_falls_back_safely(self):
        """Test 5: If the predictor encounters an error, simulation and TTC continue smoothly."""
        # Break predictor model
        corrupted_predictor = IntersectionTrajectoryPredictor(base_dir=".")
        corrupted_predictor.model = None

        tm = TrafficManager(enable_v2v_hazard=False, ai_predictor=corrupted_predictor)
        sig = SmartTrafficSignal("INT-TEST", 560, 415)

        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=700.0, speed=95.0)
        tm.ambulance = amb
        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=600.0, speed=80.0)
        tm.vehicles.append(c01)

        sim_time = 0.0
        for _ in range(40):
            sim_time += 1.0 / 60.0
            # Must not raise an exception
            tm.update(
                dt=1.0 / 60.0,
                signal_controller=sig,
                sim_time=sim_time,
                v2v_inter_manager=self.v2v,
            )

        # AI state reflects unavailable / error, but simulation and vehicles continue normally
        self.assertFalse(amb.ai_prediction_available)
        self.assertEqual(amb.ai_risk_state, "INSUFFICIENT_DATA")
        self.assertGreater(amb.speed, 0.0)

    def test_06_multiple_vehicles_maintain_independent_histories(self):
        """Test 6: Telemetry from multiple vehicles is isolated per sender ID."""
        tm = TrafficManager(enable_v2v_hazard=False, ai_predictor=self.predictor)
        sig = SmartTrafficSignal("INT-TEST", 560, 415)

        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=750.0, speed=95.0)
        tm.ambulance = amb

        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=650.0, speed=80.0)
        c02 = CivilianVehicle("C-02", "SOUTH", x=SOUTH_OVERTAKE_LANE_X, y=660.0, speed=85.0)
        tm.vehicles.extend([c01, c02])

        sim_time = 0.0
        for _ in range(45):
            sim_time += 1.0 / 60.0
            tm.update(
                dt=1.0 / 60.0,
                signal_controller=sig,
                sim_time=sim_time,
                v2v_inter_manager=self.v2v,
            )

        # Both vehicles have independent history
        self.assertGreaterEqual(self.predictor.get_history_count("C-01"), 4)
        self.assertGreaterEqual(self.predictor.get_history_count("C-02"), 4)

    def test_07_duplicate_telemetry_is_not_double_counted(self):
        """Test 7: Duplicate packets with identical timestamps are rejected by the buffer."""
        tm = TrafficManager(enable_v2v_hazard=False, ai_predictor=self.predictor)

        msg = IntersectionV2VMessage(
            sender_id="C-01",
            receiver_id="AMB-01",
            vehicle_type="CIVILIAN",
            x=592.0,
            y=650.0,
            vx=0.0,
            vy=-1.0,
            speed=80.0,
            heading=-math.pi / 2.0,
            hazard_status="NORMAL",
            timestamp=1.00,
        )

        accepted_1 = self.predictor.add_telemetry(msg)
        accepted_2 = self.predictor.add_telemetry(msg)  # Duplicate timestamp

        self.assertTrue(accepted_1)
        self.assertFalse(accepted_2)
        self.assertEqual(self.predictor.get_history_count("C-01"), 1)

    def test_08_existing_v2v_maneuver_remains_unchanged(self):
        """Test 8: Full integrated scenario with AI predictor attached preserves collision-free evasive maneuver."""
        random.seed(42)
        sig = SmartTrafficSignal("INT-TEST", 560, 415)
        tm = TrafficManager(enable_v2v_hazard=True, ai_predictor=self.predictor)
        v2i = V2IManager(v2i_range=400.0, latency=0.20)
        v2v = IntersectionV2VManager(v2v_range=200.0, latency=0.020)

        # Pre-seed ambulance approaching
        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=650.0, speed=95.0)
        amb.is_authorized = True
        tm.ambulance = amb

        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=550.0, speed=15.0)
        c01.set_hazard("DECELERATING", target_speed=15.0)
        tm.vehicles.append(c01)

        sim_time = 0.0
        evasive_initiated = False
        evasive_completed = False

        for _ in range(250):
            sim_time += 1.0 / 60.0
            sig.update(1.0 / 60.0)
            tm.update(1.0 / 60.0, sig, sim_time=sim_time, v2i_channel=v2i, rsu_pos=(560, 415), v2v_inter_manager=v2v)

            if tm.ambulance is not None:
                if tm.ambulance.is_v2v_evading:
                    evasive_initiated = True
                if tm.ambulance.v2v_evasion_complete:
                    evasive_completed = True
                    break

        self.assertTrue(evasive_initiated, "V2V evasive maneuver must initiate as in Phase 13B")
        self.assertTrue(evasive_completed, "V2V evasive lane change to Lane 2 must complete")
        self.assertAlmostEqual(tm.ambulance.x, SOUTH_OVERTAKE_LANE_X, delta=1.0)

    def test_09_event_logger_records_ai_transitions(self):
        """Test 9: EventLogger records AI lifecycle events (AI_PREDICTION_AVAILABLE)."""
        tm = TrafficManager(enable_v2v_hazard=False, ai_predictor=self.predictor)
        sig = SmartTrafficSignal("INT-TEST", 560, 415)

        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=720.0, speed=95.0)
        tm.ambulance = amb
        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=620.0, speed=80.0)
        tm.vehicles.append(c01)

        sim_time = 0.0
        for _ in range(50):
            sim_time += 1.0 / 60.0
            tm.update(1.0 / 60.0, sig, sim_time=sim_time, v2v_inter_manager=self.v2v)
            self.logger.observe_system(sim_time, sig, traffic_manager=tm)

        event_types = [e.event_type for e in self.logger.events]
        self.assertIn("AI_PREDICTION_AVAILABLE", event_types)


if __name__ == "__main__":
    unittest.main()
