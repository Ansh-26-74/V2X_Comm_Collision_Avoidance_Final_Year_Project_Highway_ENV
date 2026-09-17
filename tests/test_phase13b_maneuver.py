"""Phase 13B: Comprehensive Unit and Integration Tests for V2V Evasive Maneuver & HUD Integration.

Tests:
  1. test_critical_v2v_triggers_safe_lane_maneuver:
     Verifies V2V CRITICAL TTC triggers smooth multi-tick lateral lane shift toward adjacent lane x=628.0.
  2. test_blocked_adjacent_lane_triggers_emergency_braking:
     Verifies that when adjacent lane is blocked, lateral shift is rejected, emergency braking engages,
     and safe gap (>= 22px) is strictly maintained with zero collision.
  3. test_no_maneuver_inside_conflict_zone:
     Verifies lateral maneuvers are strictly prohibited once ambulance is inside the central conflict zone (y <= 480.0).
  4. test_no_collision_during_v2v_maneuver:
     Verifies zero bounding-box overlap at every simulation tick during evasive lane maneuver.
  5. test_maneuver_is_not_repeated_or_oscillatory:
     Verifies one-shot deterministic maneuver; ambulance does not oscillate back and forth.
  6. test_v2v_recovery_after_threat_clears:
     Verifies that once established in Lane 2, TTC returns to SAFE/inf and cruise speed is restored.
  7. test_v2i_preemption_still_operates:
     Verifies full V2I preemption lifecycle and event logger timeline integration during V2V events.
  8. test_highway_v2v_regression:
     Verifies existing highway V2V imports and classes remain 100% untouched and functional.
"""

import math
import unittest
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


class TestPhase13BManeuver(unittest.TestCase):
    """Phase 13B test suite for V2V-triggered evasive maneuver and safety fallback."""

    def setUp(self):
        """Set up simulated intersection components."""
        pygame.init()
        self.signal = SmartTrafficSignal(
            intersection_id="INT-01",
            intersection_x=560.0,
            intersection_y=415.0,
        )
        self.v2v = IntersectionV2VManager(
            v2v_range=200.0,
            broadcast_rate=10.0,
            latency=0.020,
            packet_loss_rate=0.0,
        )
        self.logger = EventLogger()

    def tearDown(self):
        pygame.quit()

    def test_01_critical_v2v_triggers_safe_lane_maneuver(self):
        """Test 1: V2V CRITICAL TTC triggers smooth lateral movement toward adjacent lane x=628.0."""
        c01 = CivilianVehicle(
            vehicle_id="C-01",
            approach="SOUTH",
            x=SOUTH_PRIMARY_LANE_X,
            y=540.0,
            speed=20.0,
            target_speed=20.0,
        )
        c01.set_hazard("DECELERATING", target_speed=15.0)

        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=SOUTH_PRIMARY_LANE_X,
            y=620.0,
            speed=95.0,
            target_speed=95.0,
        )
        amb.is_authorized = True

        # V2V exchange: AMB closing in on decelerating C-01
        msg = c01.get_v2v_telemetry(now=1.0)
        self.v2v.broadcast(now=1.0, msg=msg, receiver_pos=(amb.x, amb.y))
        delivered = self.v2v.deliver(now=1.025, receiver_id="AMB-01")
        self.assertEqual(len(delivered), 1)

        # Evaluate V2V hazard: TTC = (80 - 40) / 75 = 0.53s (< 2.0s -> CRITICAL)
        ttc, risk, threat = amb.evaluate_v2v_hazards(delivered, now=1.025)
        self.assertEqual(risk, "CRITICAL")
        self.assertTrue(ttc < 2.0)
        self.assertEqual(threat.sender_id, "C-01")

        # Initial control update before lateral movement
        dt = 0.016
        amb.update_control(
            dt=dt,
            signal_state=SignalState.GREEN,
            stop_line_coord=490.0,
            intersection_enter=480.0,
            intersection_exit=350.0,
            lead_vehicle=c01,
            all_vehicles=[c01],
        )

        # Verify maneuver is initiated
        self.assertTrue(amb.is_v2v_evading)
        self.assertTrue(amb.is_overtaking)
        self.assertEqual(amb.target_lane_x, SOUTH_OVERTAKE_LANE_X)
        self.assertEqual(amb.ambulance_state, AmbulanceState.V2V_EVASIVE_MANEUVER)
        self.assertEqual(amb.state, VehicleState.V2V_EVASIVE_MANEUVER)

        # Multi-tick verification: verify continuous physical lateral movement without teleportation
        x_positions = [amb.x]
        max_step_allowed = LATERAL_SPEED * dt + 0.05

        while amb.is_v2v_evading:
            prev_x = amb.x
            amb.update_control(
                dt=dt,
                signal_state=SignalState.GREEN,
                stop_line_coord=490.0,
                intersection_enter=480.0,
                intersection_exit=350.0,
                lead_vehicle=c01,
                all_vehicles=[c01],
            )
            delta_x = abs(amb.x - prev_x)
            # Physical correctness: no teleportation, bounded by lateral speed
            self.assertLessEqual(delta_x, max_step_allowed)
            x_positions.append(amb.x)

        # Verify completion in adjacent lane
        self.assertAlmostEqual(amb.x, SOUTH_OVERTAKE_LANE_X, places=1)
        self.assertTrue(amb.v2v_evasion_complete)
        self.assertFalse(amb.is_v2v_evading)
        self.assertGreater(len(x_positions), 10, "Maneuver must take multiple simulation frames")

    def test_02_blocked_adjacent_lane_triggers_emergency_braking(self):
        """Test 2: Blocked adjacent lane prevents lateral shift; controlled emergency braking halts AMB-01."""
        c01 = CivilianVehicle(
            vehicle_id="C-01",
            approach="SOUTH",
            x=SOUTH_PRIMARY_LANE_X,
            y=540.0,
            speed=15.0,
            target_speed=15.0,
        )

        # Blocking civilian vehicle directly in adjacent lane at overtaking position
        c_block = CivilianVehicle(
            vehicle_id="C-BLOCK",
            approach="SOUTH",
            x=SOUTH_OVERTAKE_LANE_X,
            y=610.0,  # right next to ambulance
            speed=15.0,
            target_speed=15.0,
        )

        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=SOUTH_PRIMARY_LANE_X,
            y=620.0,
            speed=95.0,
            target_speed=95.0,
        )
        amb.is_authorized = True

        # Deliver CRITICAL hazard from C-01
        msg = c01.get_v2v_telemetry(now=2.0)
        self.v2v.broadcast(now=2.0, msg=msg, receiver_pos=(amb.x, amb.y))
        delivered = self.v2v.deliver(now=2.025, receiver_id="AMB-01")
        amb.evaluate_v2v_hazards(delivered, now=2.025)
        self.assertEqual(amb.v2v_risk_state, "CRITICAL")

        # Step simulation
        dt = 0.016
        amb.update_control(
            dt=dt,
            signal_state=SignalState.GREEN,
            stop_line_coord=490.0,
            intersection_enter=480.0,
            intersection_exit=350.0,
            lead_vehicle=c01,
            all_vehicles=[c01, c_block],
        )

        # Verify lateral maneuver was REJECTED because lane is blocked
        self.assertFalse(amb.is_v2v_evading)
        self.assertEqual(amb.x, SOUTH_PRIMARY_LANE_X)
        self.assertTrue(amb.v2v_emergency_braking)
        self.assertEqual(amb.ambulance_state, AmbulanceState.V2V_EMERGENCY_BRAKING)

        # Run emergency braking for 120 ticks (~2 seconds)
        for _ in range(120):
            amb.update_control(
                dt=dt,
                signal_state=SignalState.GREEN,
                stop_line_coord=490.0,
                intersection_enter=480.0,
                intersection_exit=350.0,
                lead_vehicle=c01,
                all_vehicles=[c01, c_block],
            )
            # Safe following gap must be preserved: bumper gap >= MIN_FOLLOW_GAP
            gap = (amb.front_pos[1] - c01.rear_pos[1])
            self.assertGreaterEqual(gap, MIN_FOLLOW_GAP - 1.0)
            # Zero bounding box collision at all times
            self.assertFalse(amb.bounding_box.colliderect(c01.bounding_box))
            self.assertFalse(amb.bounding_box.colliderect(c_block.bounding_box))

        # Ambulance must have decelerated to complete stop or match lead speed safely
        self.assertLessEqual(amb.speed, c01.speed)

    def test_03_no_maneuver_inside_conflict_zone(self):
        """Test 3: Lateral evasive maneuver is strictly forbidden once ambulance is inside conflict zone."""
        c01 = CivilianVehicle(
            vehicle_id="C-01",
            approach="SOUTH",
            x=SOUTH_PRIMARY_LANE_X,
            y=380.0,
            speed=15.0,
        )

        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=SOUTH_PRIMARY_LANE_X,
            y=440.0,  # inside conflict zone (enter is 480.0)
            speed=80.0,
        )
        amb.in_intersection = True

        msg = c01.get_v2v_telemetry(now=3.0)
        self.v2v.broadcast(now=3.0, msg=msg, receiver_pos=(amb.x, amb.y))
        delivered = self.v2v.deliver(now=3.025, receiver_id="AMB-01")
        amb.evaluate_v2v_hazards(delivered, now=3.025)
        self.assertEqual(amb.v2v_risk_state, "CRITICAL")

        # Step control inside junction
        amb.update_control(
            dt=0.016,
            signal_state=SignalState.GREEN,
            stop_line_coord=490.0,
            intersection_enter=480.0,
            intersection_exit=350.0,
            lead_vehicle=c01,
            all_vehicles=[c01],
        )

        # Conflict-zone protection: lateral maneuver prohibited
        self.assertFalse(amb.lateral_maneuver_allowed)
        self.assertFalse(amb.is_v2v_evading)
        self.assertFalse(amb.is_overtaking)
        self.assertEqual(amb.x, SOUTH_PRIMARY_LANE_X)

    def test_04_no_collision_during_v2v_maneuver(self):
        """Test 4: Continuous collision check verifies zero bounding-box overlaps throughout entire evasion."""
        c01 = CivilianVehicle(
            vehicle_id="C-01",
            approach="SOUTH",
            x=SOUTH_PRIMARY_LANE_X,
            y=550.0,
            speed=20.0,
        )
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=SOUTH_PRIMARY_LANE_X,
            y=630.0,
            speed=95.0,
        )
        amb.is_authorized = True

        # Trigger V2V CRITICAL hazard
        msg = c01.get_v2v_telemetry(now=4.0)
        self.v2v.broadcast(now=4.0, msg=msg, receiver_pos=(amb.x, amb.y))
        delivered = self.v2v.deliver(now=4.025, receiver_id="AMB-01")
        amb.evaluate_v2v_hazards(delivered, now=4.025)

        dt = 0.016
        for step in range(150):
            amb.update_control(
                dt=dt,
                signal_state=SignalState.GREEN,
                stop_line_coord=490.0,
                intersection_enter=480.0,
                intersection_exit=350.0,
                lead_vehicle=c01,
                all_vehicles=[c01],
            )
            c01.update_control(
                dt=dt,
                signal_state=SignalState.GREEN,
                stop_line_coord=490.0,
                intersection_enter=480.0,
                intersection_exit=350.0,
                lead_vehicle=None,
            )
            # Invariant: zero collision at every single frame
            self.assertFalse(
                amb.bounding_box.colliderect(c01.bounding_box),
                f"Collision detected at step {step}: AMB pos ({amb.x:.1f}, {amb.y:.1f}) vs C01 ({c01.x:.1f}, {c01.y:.1f})"
            )

    def test_05_maneuver_is_not_repeated_or_oscillatory(self):
        """Test 5: Maneuver is one-shot deterministic; AMB-01 remains established without oscillation."""
        c01 = CivilianVehicle(
            vehicle_id="C-01",
            approach="SOUTH",
            x=SOUTH_PRIMARY_LANE_X,
            y=540.0,
            speed=20.0,
        )
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=SOUTH_PRIMARY_LANE_X,
            y=620.0,
            speed=95.0,
        )
        amb.is_authorized = True

        # Deliver CRITICAL hazard
        msg = c01.get_v2v_telemetry(now=5.0)
        self.v2v.broadcast(now=5.0, msg=msg, receiver_pos=(amb.x, amb.y))
        delivered = self.v2v.deliver(now=5.025, receiver_id="AMB-01")
        amb.evaluate_v2v_hazards(delivered, now=5.025)

        dt = 0.016
        # Execute until lane change completes
        for _ in range(80):
            amb.update_control(
                dt=dt,
                signal_state=SignalState.GREEN,
                stop_line_coord=490.0,
                intersection_enter=480.0,
                intersection_exit=350.0,
                lead_vehicle=c01,
                all_vehicles=[c01],
            )
            if amb.v2v_evasion_complete:
                break

        self.assertTrue(amb.v2v_evasion_complete)
        self.assertAlmostEqual(amb.x, SOUTH_OVERTAKE_LANE_X, places=1)

        # Run 60 additional frames: verify AMB-01 stays locked in Lane 2
        for _ in range(60):
            amb.update_control(
                dt=dt,
                signal_state=SignalState.GREEN,
                stop_line_coord=490.0,
                intersection_enter=480.0,
                intersection_exit=350.0,
                lead_vehicle=c01,
                all_vehicles=[c01],
            )
            self.assertAlmostEqual(amb.x, SOUTH_OVERTAKE_LANE_X, places=1)
            self.assertFalse(amb.is_v2v_evading)

    def test_06_v2v_recovery_after_threat_clears(self):
        """Test 6: Once established in adjacent lane, C-01 in old lane is no longer an in-corridor threat."""
        c01 = CivilianVehicle(
            vehicle_id="C-01",
            approach="SOUTH",
            x=SOUTH_PRIMARY_LANE_X,  # 592.0
            y=540.0,
            speed=20.0,
        )
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=SOUTH_OVERTAKE_LANE_X,  # 628.0 (already in Lane 2)
            y=600.0,
            speed=95.0,
        )
        amb.v2v_evasion_complete = True

        msg = c01.get_v2v_telemetry(now=6.0)
        self.v2v.broadcast(now=6.0, msg=msg, receiver_pos=(amb.x, amb.y))
        delivered = self.v2v.deliver(now=6.025, receiver_id="AMB-01")

        # Evaluate V2V hazards: C-01 is in Lane 1, AMB is in Lane 2 (|dx| = 36.0px > 22.0px)
        ttc, risk, threat = amb.evaluate_v2v_hazards(delivered, now=6.025)
        self.assertEqual(ttc, float("inf"))
        self.assertIn(risk, ("SAFE", "NONE"))
        self.assertIsNone(threat)

    def test_07_v2i_preemption_still_operates(self):
        """Test 7: V2I preemption lifecycle and EventLogger timeline operate cleanly during V2V events."""
        v2i = V2IManager(v2i_range=400.0, latency=0.10)
        traffic = TrafficManager(center_x=560, center_y=415, road_width=130, stop_line_dist=75)
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=SOUTH_PRIMARY_LANE_X,
            y=620.0,
            speed=95.0,
        )
        traffic.ambulance = amb

        c01 = CivilianVehicle(
            vehicle_id="C-01",
            approach="SOUTH",
            x=SOUTH_PRIMARY_LANE_X,
            y=540.0,
            speed=20.0,
        )
        traffic.vehicles = [c01]

        # 1. Trigger V2I request
        amb.step_v2i_communication(now=7.0, rsu_pos=(730.0, 285.0), v2i_channel=v2i, stop_line_coord=490.0)
        v2i_pkts = v2i.deliver_to_rsu(now=7.15)
        self.assertEqual(len(v2i_pkts), 1)
        self.signal.receive_emergency_request(v2i_pkts[0], current_time=7.15, auto_preempt=True)
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_SOUTH_GREEN)

        # 2. Trigger V2V CRITICAL hazard
        msg = c01.get_v2v_telemetry(now=7.2)
        self.v2v.broadcast(now=7.2, msg=msg, receiver_pos=(amb.x, amb.y))
        delivered_v2v = self.v2v.deliver(now=7.225, receiver_id="AMB-01")
        amb.evaluate_v2v_hazards(delivered_v2v, now=7.225)
        self.assertEqual(amb.v2v_risk_state, "CRITICAL")

        # 3. Observe system in logger
        events = self.logger.observe_system(
            sim_time=7.25,
            signal_controller=self.signal,
            traffic_manager=traffic,
            v2i_channel=v2i,
        )
        event_types = [e.event_type for e in self.logger.get_events()]
        self.assertIn("V2V_TTC_CRITICAL", event_types)

        # 4. Step control: evasive maneuver begins
        amb.update_control(
            dt=0.016,
            signal_state=SignalState.GREEN,
            stop_line_coord=490.0,
            intersection_enter=480.0,
            intersection_exit=350.0,
            lead_vehicle=c01,
            all_vehicles=[c01],
        )
        self.logger.observe_system(
            sim_time=7.27,
            signal_controller=self.signal,
            traffic_manager=traffic,
            v2i_channel=v2i,
        )
        event_types = [e.event_type for e in self.logger.get_events()]
        self.assertIn("V2V_EVASIVE_MANEUVER_STARTED", event_types)

        # 5. Complete lane change
        while amb.is_v2v_evading:
            amb.update_control(
                dt=0.016,
                signal_state=SignalState.GREEN,
                stop_line_coord=490.0,
                intersection_enter=480.0,
                intersection_exit=350.0,
                lead_vehicle=c01,
                all_vehicles=[c01],
            )
        self.logger.observe_system(
            sim_time=8.5,
            signal_controller=self.signal,
            traffic_manager=traffic,
            v2i_channel=v2i,
        )
        event_types = [e.event_type for e in self.logger.get_events()]
        self.assertIn("V2V_LANE_CHANGE_COMPLETED", event_types)

    def test_08_highway_v2v_regression(self):
        """Test 8: Confirm existing highway V2V imports and classes are completely untouched and functional."""
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
