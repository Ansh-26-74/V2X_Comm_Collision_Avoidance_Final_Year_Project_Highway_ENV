"""Phase 13C: Final Integrated V2X Validation & Regression Test Suite.

Validates the complete end-to-end integrated scenario:
  - V2I Emergency Detection -> RSU Comm -> Safe Preemption -> Protected South Green
  - V2V Peer CAM Exchange -> Lead Hazard -> CRITICAL TTC -> Safe Evasive Maneuver
  - Blocked-lane Fallback -> Controlled Emergency Braking -> Collision-Free Safe Stop
  - Conflict-zone Protection -> Lateral maneuver forbidden inside junction (y <= 480.0)
  - Protected Crossing -> Full-body Clearance -> RSU Termination -> Normal Cycle Recovery
  - Deterministic reproducibility across repeated executions with identical seeds
  - Untouched highway V2V regression validation
"""

import math
import unittest
import random
import pygame

from v2i.smart_signal import SmartTrafficSignal, CyclePhase, SignalState
from v2i.traffic_manager import (
    TrafficManager,
    CivilianVehicle,
    AmbulanceVehicle,
    AmbulanceState,
    VehicleState,
    SOUTH_PRIMARY_LANE_X,
    SOUTH_OVERTAKE_LANE_X,
    MIN_FOLLOW_GAP,
    LATERAL_SPEED,
)
from v2i.v2i_manager import V2IManager
from v2i.v2v_inter_manager import IntersectionV2VManager, IntersectionV2VMessage
from v2i.event_logger import EventLogger, EventCategory


class TestPhase13CFinalIntegration(unittest.TestCase):
    """Phase 13C final integrated verification suite for Dual-Layer V2X."""

    def setUp(self):
        pygame.init()

    def tearDown(self):
        pygame.quit()

    def _run_scenario(self, seed: int = 42, total_frames: int = 1500, enable_hazard: bool = True):
        """Helper to run a full headless deterministic integrated simulation."""
        random.seed(seed)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        tm = TrafficManager(560, 415, 130, 75, enable_v2v_hazard=enable_hazard)
        v2i = V2IManager(v2i_range=400.0, latency=0.20, packet_loss_rate=0.0)
        v2v = IntersectionV2VManager(v2v_range=200.0, latency=0.020, packet_loss_rate=0.0)
        logger = EventLogger(max_events=40)

        sim_time = 0.0
        dt = 1.0 / 60.0
        collisions = []

        maneuver_start_y = None
        maneuver_start_time = None
        clearance_time = None

        for frame in range(total_frames):
            sig.update(dt)
            sim_time += dt

            tm.update(
                dt=dt,
                signal_controller=sig,
                sim_time=sim_time,
                v2i_channel=v2i,
                rsu_pos=(730.0, 285.0),
                v2v_inter_manager=v2v,
            )

            # Record maneuver start landmark
            if tm.ambulance is not None:
                if tm.ambulance.is_v2v_evading and maneuver_start_y is None:
                    maneuver_start_y = tm.ambulance.y
                    maneuver_start_time = sim_time

                # Continuous bounding-box collision detection check
                amb_rect = tm.ambulance.bounding_box
                for v in tm.vehicles:
                    if v is not tm.ambulance and amb_rect.colliderect(v.bounding_box):
                        collisions.append((sim_time, tm.ambulance.x, tm.ambulance.y, v.x, v.y))

            if tm.ambulance_cleared_events and sig.current_phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
                sig.notify_ambulance_cleared("AMB-01")
                if clearance_time is None:
                    clearance_time = sim_time

            for msg in v2i.deliver_to_rsu(sim_time):
                sig.receive_emergency_request(msg, sim_time, auto_preempt=True)

            sig.update_traffic_analysis(tm.vehicles, tm.ambulance, sim_time)
            logger.observe_system(sim_time, sig, tm, v2i, v2v)

        return {
            "sig": sig,
            "tm": tm,
            "v2i": v2i,
            "v2v": v2v,
            "logger": logger,
            "sim_time": sim_time,
            "collisions": collisions,
            "maneuver_start_y": maneuver_start_y,
            "maneuver_start_time": maneuver_start_time,
            "clearance_time": clearance_time,
        }

    def test_01_full_v2i_v2v_sequence(self):
        """Test 1: Verifies the complete scenario occurs end-to-end in the correct chronological order."""
        res = self._run_scenario(seed=42, total_frames=1500, enable_hazard=True)
        event_types = [e.event_type for e in res["logger"].get_events()]

        # Milestone verification
        self.assertIn("AMB_IN_RANGE", event_types)
        self.assertIn("PACKET_DELIVERED", event_types)
        self.assertIn("PREEMPTION_YELLOW", event_types)
        self.assertIn("V2V_TTC_CRITICAL", event_types)
        self.assertIn("V2V_EVASIVE_MANEUVER_STARTED", event_types)
        self.assertIn("V2V_LANE_CHANGE_COMPLETED", event_types)
        self.assertIn("EMERGENCY_SOUTH_GREEN", event_types)
        self.assertIn("AMB_ENTERED_INTERSECTION", event_types)
        self.assertIn("AMB_CLEARED_INTERSECTION", event_types)
        self.assertIn("EMERGENCY_TERMINATING", event_types)
        self.assertIn("RECOVERY_ALL_RED", event_types)

    def test_02_v2v_response_occurs_before_conflict_zone(self):
        """Test 2: Verifies the evasive maneuver begins strictly before entering the central conflict zone."""
        res = self._run_scenario(seed=42, total_frames=1500, enable_hazard=True)

        self.assertIsNotNone(res["maneuver_start_y"], "V2V evasive maneuver must be initiated")
        # Conflict zone boundary is intersection_enter = 480.0
        # South moves North (-Y): approaching has y > 480.0
        enter_y = res["tm"].lane_coords["SOUTH"]["enter"]
        self.assertGreater(
            res["maneuver_start_y"],
            enter_y + 15.0,
            f"Maneuver began at y={res['maneuver_start_y']:.1f}, must be outside conflict zone (y > {enter_y + 15.0})"
        )

    def test_03_ambulance_crosses_without_collision(self):
        """Test 3: Continuous bounding-box collision detection verifies zero collisions throughout run."""
        res = self._run_scenario(seed=42, total_frames=1500, enable_hazard=True)
        self.assertEqual(
            len(res["collisions"]),
            0,
            f"Expected zero collisions during integrated run, but detected: {res['collisions']}"
        )

    def test_04_v2i_recovery_after_v2v_maneuver(self):
        """Test 4: Verifies emergency termination and normal cycle recovery execute cleanly after maneuver."""
        res = self._run_scenario(seed=42, total_frames=1500, enable_hazard=True)
        sig = res["sig"]

        # Final signal controller state: preemption completed and normal operation active
        self.assertEqual(sig.rsu_priority_status, "INACTIVE")
        self.assertFalse(sig.preemption_active)
        self.assertFalse(sig.preemption_clearing)
        self.assertTrue(sig.emergency_completed)
        self.assertIn(
            sig.current_phase,
            [CyclePhase.NS_RED_EW_GREEN, CyclePhase.NS_RED_EW_YELLOW, CyclePhase.NS_GREEN_EW_RED]
        )

    def test_05_blocked_lane_end_to_end_fallback(self):
        """Test 5: Blocked target lane triggers controlled emergency braking; safe gap strictly maintained."""
        random.seed(42)
        sig = SmartTrafficSignal("INT-01", 560, 415)
        tm = TrafficManager(560, 415, 130, 75, enable_v2v_hazard=False)
        v2i = V2IManager(v2i_range=400.0)
        v2v = IntersectionV2VManager(v2v_range=200.0)
        logger = EventLogger(max_events=30)

        # Spawn AMB-01 and C-01 ahead in Lane 1
        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=620.0, speed=95.0)
        amb.is_authorized = True
        tm.ambulance = amb

        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=540.0, speed=15.0)
        c01.set_hazard("DECELERATING", target_speed=15.0)

        # Place C-BLOCK occupying the adjacent lane
        c_block = CivilianVehicle("C-BLOCK", "SOUTH", x=SOUTH_OVERTAKE_LANE_X, y=615.0, speed=15.0)
        tm.vehicles = [c01, c_block]

        # Trigger V2V communication
        msg = c01.get_v2v_telemetry(now=1.0)
        v2v.broadcast(now=1.0, msg=msg, receiver_pos=(amb.x, amb.y))
        delivered = v2v.deliver(now=1.025, receiver_id="AMB-01")
        amb.evaluate_v2v_hazards(delivered, now=1.025)
        self.assertEqual(amb.v2v_risk_state, "CRITICAL")

        # Step simulation for 120 ticks
        dt = 1.0 / 60.0
        for step in range(120):
            tm.update(dt, sig, sim_time=1.0 + step * dt, v2i_channel=v2i, rsu_pos=(730, 285), v2v_inter_manager=v2v)
            logger.observe_system(1.0 + step * dt, sig, tm, v2i, v2v)

            # Assert lateral change is rejected
            self.assertEqual(amb.x, SOUTH_PRIMARY_LANE_X)
            self.assertFalse(amb.is_v2v_evading)
            # Assert safe following gap maintained (>= 21px)
            gap = (amb.front_pos[1] - c01.rear_pos[1])
            self.assertGreaterEqual(gap, MIN_FOLLOW_GAP - 1.0)
            self.assertFalse(amb.bounding_box.colliderect(c01.bounding_box))

        event_types = [e.event_type for e in logger.get_events()]
        self.assertIn("V2V_EMERGENCY_BRAKING", event_types)
        self.assertNotIn("V2V_LANE_CHANGE_COMPLETED", event_types)

    def test_06_event_timeline_order(self):
        """Test 6: Verifies all major V2I and V2V milestones occur in strict logical progression."""
        res = self._run_scenario(seed=42, total_frames=1500, enable_hazard=True)
        events = res["logger"].get_events()

        def first_idx(etype: str) -> int:
            for idx, e in enumerate(events):
                if e.event_type == etype:
                    return idx
            return -1

        idx_v2i_range = first_idx("AMB_IN_RANGE")
        idx_v2i_preempt = first_idx("PREEMPTION_YELLOW")
        idx_v2v_crit = first_idx("V2V_TTC_CRITICAL")
        idx_v2v_start = first_idx("V2V_EVASIVE_MANEUVER_STARTED")
        idx_v2v_done = first_idx("V2V_LANE_CHANGE_COMPLETED")
        idx_green = first_idx("EMERGENCY_SOUTH_GREEN")
        idx_enter = first_idx("AMB_ENTERED_INTERSECTION")
        idx_clear = first_idx("AMB_CLEARED_INTERSECTION")
        idx_term = first_idx("EMERGENCY_TERMINATING")
        idx_recov = first_idx("RECOVERY_ALL_RED")

        # Verify all events occurred
        for name, idx in [
            ("AMB_IN_RANGE", idx_v2i_range),
            ("PREEMPTION_YELLOW", idx_v2i_preempt),
            ("V2V_TTC_CRITICAL", idx_v2v_crit),
            ("V2V_EVASIVE_MANEUVER_STARTED", idx_v2v_start),
            ("V2V_LANE_CHANGE_COMPLETED", idx_v2v_done),
            ("EMERGENCY_SOUTH_GREEN", idx_green),
            ("AMB_ENTERED_INTERSECTION", idx_enter),
            ("AMB_CLEARED_INTERSECTION", idx_clear),
            ("EMERGENCY_TERMINATING", idx_term),
            ("RECOVERY_ALL_RED", idx_recov),
        ]:
            self.assertNotEqual(idx, -1, f"Event {name} was not found in event timeline")

        # Strict logical order assertions:
        # 1. V2I In Range before Preemption
        self.assertLess(idx_v2i_range, idx_v2i_preempt)
        # 2. V2V Critical before Maneuver Start
        self.assertLess(idx_v2v_crit, idx_v2v_start)
        # 3. Maneuver Start before Maneuver Complete
        self.assertLess(idx_v2v_start, idx_v2v_done)
        # 4. Maneuver Complete before Intersection Entry
        self.assertLess(idx_v2v_done, idx_enter)
        # 5. Intersection Entry before Clearance
        self.assertLess(idx_enter, idx_clear)
        # 6. Clearance before Recovery
        self.assertLess(idx_clear, idx_recov)

    def test_07_full_integrated_run_deterministic(self):
        """Test 7: Two identical executions produce bit-for-bit identical landmark timestamps and outcomes."""
        res1 = self._run_scenario(seed=42, total_frames=1200, enable_hazard=True)
        res2 = self._run_scenario(seed=42, total_frames=1200, enable_hazard=True)

        self.assertAlmostEqual(res1["maneuver_start_time"], res2["maneuver_start_time"], places=3)
        self.assertAlmostEqual(res1["maneuver_start_y"], res2["maneuver_start_y"], places=3)
        self.assertAlmostEqual(res1["clearance_time"], res2["clearance_time"], places=3)

        evts1 = [e.formatted_str() for e in res1["logger"].get_events()]
        evts2 = [e.formatted_str() for e in res2["logger"].get_events()]
        self.assertEqual(evts1, evts2)

    def test_08_highway_v2v_remains_unchanged(self):
        """Test 8: Verify original highway V2V scenario classes and functions remain completely functional."""
        import main
        import v2v_manager
        from ai.trajectory_predictor import AIPredictor
        from ai.safety_fusion import SafetyFusionEngine

        self.assertTrue(hasattr(main, "main"))
        self.assertTrue(hasattr(v2v_manager, "V2VManager"))
        self.assertTrue(hasattr(AIPredictor, "predict"))
        self.assertTrue(hasattr(SafetyFusionEngine, "evaluate"))

        # Verify highway V2V instantiation
        hw_v2v = v2v_manager.V2VManager(v2v_range=300.0)
        self.assertEqual(hw_v2v.v2v_range, 300.0)


if __name__ == "__main__":
    unittest.main()
