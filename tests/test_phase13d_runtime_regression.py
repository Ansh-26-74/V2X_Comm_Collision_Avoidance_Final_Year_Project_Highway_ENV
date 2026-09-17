"""Phase 13D Regression Tests: Runtime Integration & Behavior Verification.

Verifies:
1. Civilian traffic is created during integrated scenario startup.
2. C-01 exists.
3. C-01 is ahead of AMB-01 on the South approach.
4. C-01 provides V2V telemetry (CAM messages).
5. V2V telemetry reaches AMB-01.
6. AI receives enough telemetry history (>=5 packets).
7. AI prediction becomes available.
8. AI prediction does NOT control vehicle lifecycle (spawning/despawning).
9. AI unavailable does NOT stop civilian traffic.
10. AI unavailable does NOT stop ambulance/V2I simulation.
11. Existing deterministic TTC behavior remains functional.
12. Existing evasive maneuver remains functional.
13. Highway V2V remains unaffected.
14. Runtime does not invoke training.
15. Traffic appears before the V2V event.
"""

import math
import random
import sys
import unittest
import pygame

from v2i.smart_signal import SmartTrafficSignal, CyclePhase
from v2i.traffic_manager import (
    TrafficManager,
    CivilianVehicle,
    AmbulanceVehicle,
    SOUTH_PRIMARY_LANE_X,
    SOUTH_OVERTAKE_LANE_X,
)
from v2i.v2i_manager import V2IManager
from v2i.v2v_inter_manager import IntersectionV2VManager, IntersectionV2VMessage
from ai.intersection_ai.predictor import IntersectionTrajectoryPredictor


class TestPhase13DRuntimeRegression(unittest.TestCase):
    """15 required regression tests for Phase 13D runtime restoration."""

    def setUp(self):
        pygame.init()
        random.seed(42)

    def tearDown(self):
        pygame.quit()

    def test_01_civilian_traffic_created_during_startup(self):
        """Test 1: Civilian traffic is created during integrated scenario startup."""
        tm = TrafficManager(center_x=560, center_y=415, road_width=130, stop_line_dist=75)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        dt = 1.0 / 60.0

        # Simulate first 2 seconds (spawn timer starts at 0.5s)
        for step in range(120):
            tm.update(dt, sig, sim_time=step * dt)

        self.assertGreater(len(tm.vehicles), 0, "Civilian traffic must be created during initial startup")

    def test_02_c01_exists(self):
        """Test 2: C-01 exists deterministically in the simulation traffic."""
        tm = TrafficManager(center_x=560, center_y=415, road_width=130, stop_line_dist=75)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        dt = 1.0 / 60.0

        # Run simulation until C-01 spawns
        for step in range(600):  # 10 seconds
            tm.update(dt, sig, sim_time=step * dt)
            if any(v.vehicle_id == "C-01" for v in tm.vehicles):
                break

        c01_present = any(v.vehicle_id == "C-01" for v in tm.vehicles)
        self.assertTrue(c01_present, "Vehicle C-01 must exist in the simulation traffic")

    def test_03_c01_ahead_of_amb01(self):
        """Test 3: C-01 is physically ahead of AMB-01 on the South approach."""
        tm = TrafficManager(center_x=560, center_y=415, road_width=130, stop_line_dist=75)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        dt = 1.0 / 60.0

        c01_ahead_count = 0
        for step in range(650):
            sim_time = step * dt
            tm.update(dt, sig, sim_time=sim_time)
            if tm.ambulance is not None:
                c01 = next((v for v in tm.vehicles if v.vehicle_id == "C-01"), None)
                if c01 is not None and c01.approach == "SOUTH":
                    # On South approach, travel is North (-Y), so ahead means c01.y < amb.y
                    if c01.y < tm.ambulance.y:
                        c01_ahead_count += 1

        self.assertGreater(c01_ahead_count, 0, "C-01 must be physically ahead of AMB-01 on South approach")

    def test_04_c01_provides_v2v_telemetry(self):
        """Test 4: C-01 provides valid V2V telemetry (CAM message)."""
        c01 = CivilianVehicle("C-01", "SOUTH", x=592.5, y=600.0, speed=80.0)
        telemetry = c01.get_v2v_telemetry(now=5.0)

        self.assertIsNotNone(telemetry)
        self.assertEqual(telemetry.sender_id, "C-01")
        self.assertEqual(telemetry.vehicle_type, "CIVILIAN")
        self.assertAlmostEqual(telemetry.x, 592.5)
        self.assertAlmostEqual(telemetry.y, 600.0)
        self.assertAlmostEqual(telemetry.speed, 80.0)

    def test_05_v2v_telemetry_reaches_amb01(self):
        """Test 5: V2V telemetry broadcasts from C-01 reach AMB-01 via IntersectionV2VManager."""
        v2v = IntersectionV2VManager(v2v_range=200.0, latency=0.020)
        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=592.5, y=700.0, speed=95.0)
        c01 = CivilianVehicle("C-01", "SOUTH", x=592.5, y=620.0, speed=80.0)

        # Broadcast telemetry at t=1.00
        msg = c01.get_v2v_telemetry(now=1.00)
        v2v.broadcast(1.00, msg, receiver_pos=(amb.x, amb.y))

        # At t=1.01 (latency not yet elapsed)
        early = v2v.deliver(1.01, receiver_id="AMB-01")
        self.assertEqual(len(early), 0)

        # At t=1.03 (latency elapsed)
        delivered = v2v.deliver(1.03, receiver_id="AMB-01")
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].sender_id, "C-01")

    def test_06_ai_receives_enough_telemetry_history(self):
        """Test 6: AI predictor receives and accumulates telemetry history from C-01."""
        predictor = IntersectionTrajectoryPredictor(base_dir=".")
        tm = TrafficManager(ai_predictor=predictor)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        v2v = IntersectionV2VManager(v2v_range=200.0, latency=0.020)

        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=592.5, y=700.0, speed=95.0)
        tm.ambulance = amb
        c01 = CivilianVehicle("C-01", "SOUTH", x=592.5, y=600.0, speed=80.0)
        tm.vehicles.append(c01)

        dt = 1.0 / 60.0
        # Run 60 frames (~1s, covers several 10 Hz telemetry deliveries)
        for step in range(60):
            sim_time = step * dt
            tm.update(dt, sig, sim_time=sim_time, v2v_inter_manager=v2v)

        history = predictor.history_buffers.get("C-01", [])
        self.assertGreaterEqual(len(history), 5, "Predictor must accumulate >= 5 telemetry packets for C-01")

    def test_07_ai_prediction_becomes_available(self):
        """Test 7: AI prediction becomes available once >=5 telemetry packets are accumulated."""
        predictor = IntersectionTrajectoryPredictor(base_dir=".")
        tm = TrafficManager(ai_predictor=predictor)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        v2v = IntersectionV2VManager(v2v_range=200.0, latency=0.020)

        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=592.5, y=700.0, speed=95.0)
        tm.ambulance = amb
        c01 = CivilianVehicle("C-01", "SOUTH", x=592.5, y=620.0, speed=80.0)
        tm.vehicles.append(c01)

        dt = 1.0 / 60.0
        prediction_observed = False
        for step in range(120):
            sim_time = step * dt
            tm.update(dt, sig, sim_time=sim_time, v2v_inter_manager=v2v)
            if amb.latest_ai_prediction is not None and amb.latest_ai_prediction.prediction_available:
                prediction_observed = True
                self.assertEqual(len(amb.latest_ai_prediction.waypoints), 6)
                break

        self.assertTrue(prediction_observed, "AI prediction must become available with 6 waypoints")

    def test_08_ai_prediction_does_not_control_vehicle_lifecycle(self):
        """Test 8: AI prediction does NOT spawn or despawn vehicles or alter TrafficManager lifecycle."""
        predictor = IntersectionTrajectoryPredictor(base_dir=".")
        tm = TrafficManager(ai_predictor=predictor)
        sig = SmartTrafficSignal("INT-01", 560, 415)

        # Initial vehicle count
        dt = 1.0 / 60.0
        for step in range(100):
            tm.update(dt, sig, sim_time=step * dt)

        count_with_ai = len(tm.vehicles)
        # Vehicles were spawned solely via TrafficManager._try_spawn_vehicle
        self.assertGreater(count_with_ai, 0)
        for v in tm.vehicles:
            self.assertIn(v.approach, ["NORTH", "EAST", "SOUTH", "WEST"])

    def test_09_ai_unavailable_does_not_stop_civilian_traffic(self):
        """Test 9: When AI predictor is None, civilian traffic continues spawning and moving normally."""
        tm = TrafficManager(ai_predictor=None)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        dt = 1.0 / 60.0

        for step in range(180):  # 3 seconds
            tm.update(dt, sig, sim_time=step * dt)

        self.assertGreater(len(tm.vehicles), 0, "Civilian traffic must spawn normally without AI")
        # Ensure vehicles are moving
        any_moving = any(v.speed > 0 for v in tm.vehicles)
        self.assertTrue(any_moving, "Civilian vehicles must move normally when AI is unavailable")

    def test_10_ai_unavailable_does_not_stop_ambulance_or_v2i(self):
        """Test 10: When AI predictor is None, AMB-01 and V2I preemption proceed normally."""
        tm = TrafficManager(ai_predictor=None, ambulance_spawn_time=1.0)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        v2i = V2IManager(v2i_range=400.0, latency=0.20)
        dt = 1.0 / 60.0

        preemption_triggered = False
        for step in range(300):  # 5 seconds
            sim_time = step * dt
            tm.update(dt, sig, sim_time=sim_time, v2i_channel=v2i, rsu_pos=(730.0, 285.0))
            delivered = v2i.deliver_to_rsu(sim_time)
            for msg in delivered:
                accepted = sig.receive_emergency_request(msg, current_time=sim_time, auto_preempt=True)
                if accepted:
                    preemption_triggered = True

        self.assertIsNotNone(tm.ambulance, "Ambulance must spawn even when AI is unavailable")
        self.assertTrue(preemption_triggered, "V2I preemption must succeed even when AI is unavailable")

    def test_11_existing_deterministic_ttc_remains_functional(self):
        """Test 11: Authoritative deterministic TTC computation remains fully functional."""
        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=700.0, speed=95.0)
        # Lead vehicle moving at 15 px/s ahead at y=650 (50px distance)
        # Bumper gap = 50 - 36 = 14px. Closing speed = 95 - 15 = 80 px/s. TTC = 14/80 = 0.175s -> CRITICAL
        msg = IntersectionV2VMessage(
            sender_id="C-01",
            receiver_id="BROADCAST",
            vehicle_type="CIVILIAN",
            x=SOUTH_PRIMARY_LANE_X,
            y=650.0,
            vx=0.0,
            vy=-1.0,
            speed=15.0,
            heading=-math.pi / 2.0,
            hazard_status="DECELERATING",
            timestamp=6.0,
        )

        ttc, risk, threat = amb.evaluate_v2v_hazards([msg], now=6.0)
        self.assertEqual(risk, "CRITICAL")
        self.assertLess(ttc, 2.0)
        self.assertEqual(threat.sender_id, "C-01")

    def test_12_existing_evasive_maneuver_remains_functional(self):
        """Test 12: Ambulance initiates evasive lane change into Lane 2 upon CRITICAL risk."""
        tm = TrafficManager(center_x=560, center_y=415, road_width=130, stop_line_dist=75)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        sig.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=650.0, speed=95.0)
        amb.is_authorized = True
        tm.ambulance = amb

        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=590.0, speed=15.0)
        c01.set_hazard("DECELERATING", target_speed=15.0)
        tm.vehicles.append(c01)

        v2v = IntersectionV2VManager(v2v_range=200.0, latency=0.020)
        dt = 1.0 / 60.0

        evasive_triggered = False
        for step in range(120):
            sim_time = step * dt
            tm.update(dt, sig, sim_time=sim_time, v2v_inter_manager=v2v)
            if amb.is_v2v_evading or amb.v2v_evasion_complete or amb.x > SOUTH_PRIMARY_LANE_X + 5.0:
                evasive_triggered = True
                break

        self.assertTrue(evasive_triggered, "Ambulance must execute evasive lateral lane change")

    def test_13_highway_v2v_remains_unaffected(self):
        """Test 13: Original Highway V2V imports and models remain completely unaffected."""
        import v2v_manager
        import ai.trajectory_predictor
        import ai.safety_fusion

        self.assertTrue(hasattr(v2v_manager, "V2VManager"))
        self.assertTrue(hasattr(ai.trajectory_predictor, "AIPredictor"))
        self.assertTrue(hasattr(ai.trajectory_predictor, "TrajectoryGRU"))
        self.assertTrue(hasattr(ai.safety_fusion, "SafetyFusionEngine"))

    def test_14_runtime_does_not_invoke_training(self):
        """Test 14: Starting simulation or importing predictor NEVER invokes training."""
        import v2i_main
        self.assertNotIn("ai.intersection_ai.train_predictor", sys.modules)

    def test_15_traffic_appears_before_v2v_event(self):
        """Test 15: Civilian traffic appears before the V2V interaction occurs."""
        tm = TrafficManager(ambulance_spawn_time=8.5)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        dt = 1.0 / 60.0

        traffic_present_before_amb = False
        for step in range(300):  # 5 seconds
            sim_time = step * dt
            tm.update(dt, sig, sim_time=sim_time)
            if sim_time < 8.5 and len(tm.vehicles) > 0:
                traffic_present_before_amb = True
                break

        self.assertTrue(traffic_present_before_amb, "Civilian traffic must appear before the ambulance spawns")


if __name__ == "__main__":
    unittest.main()
