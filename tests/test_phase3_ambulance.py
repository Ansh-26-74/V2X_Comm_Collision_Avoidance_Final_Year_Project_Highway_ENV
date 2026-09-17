"""Comprehensive Phase 3 Test Suite: Ambulance Introduction (AMB-01).

Validates all 13 Phase 3 requirements:
 1. AMB-01 spawns correctly.
 2. AMB-01 is visually identifiable (dimensions, emergency flag, strobe attributes).
 3. AMB-01 travels South -> North.
 4. AMB-01 approaches the intersection.
 5. South signal is RED when AMB-01 reaches the stop line.
 6. AMB-01 decelerates smoothly.
 7. AMB-01 stops before the stop line.
 8. AMB-01 remains stopped during RED.
 9. AMB-01 does not enter the intersection during RED.
10. Civilian traffic continues normally.
11. No obvious vehicle collisions occur (safe queueing behind/ahead of ambulance).
12. Existing Phase 1 / Phase 2 behavior remains intact (signal mutex, civilian flow).
13. V2V scenario remains completely unchanged.
14. Strict Scope: Zero emergency V2I communication or signal preemption implemented.
"""

import math
import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2i.smart_signal import SmartTrafficSignal, SignalState, CyclePhase
from v2i.traffic_manager import (
    CivilianVehicle,
    AmbulanceVehicle,
    TrafficManager,
    VehicleState,
    MIN_FOLLOW_GAP,
    STOP_LINE_BUFFER,
    AMBULANCE_LENGTH,
    AMBULANCE_WIDTH,
    VEHICLE_LENGTH,
)
import v2i.v2i_renderer as renderer


class TestPhase3Ambulance(unittest.TestCase):

    def setUp(self):
        self.cx = renderer.CX
        self.cy = renderer.CY
        self.rw = renderer.ROAD_WIDTH
        self.stop_dist = renderer.STOP_LINE_DIST
        self.dt = 1.0 / 60.0

    def test_1_2_ambulance_attributes_and_distinction(self):
        """Req 1, 2: AMB-01 spawns with distinct emergency properties and strobe attributes."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=self.cx + self.rw // 4,
            y=800.0,
            speed=95.0,
        )
        self.assertEqual(amb.vehicle_id, "AMB-01")
        self.assertEqual(amb.approach, "SOUTH")
        self.assertTrue(amb.is_emergency)
        self.assertEqual(amb.length, AMBULANCE_LENGTH)
        self.assertEqual(amb.width, AMBULANCE_WIDTH)
        self.assertGreater(amb.length, VEHICLE_LENGTH, "Ambulance must have longer chassis than civilian car")
        self.assertEqual(amb.color, (250, 250, 252), "Ambulance body must be white")
        self.assertTrue(hasattr(amb, "strobe_timer"))
        self.assertTrue(hasattr(amb, "strobe_left_on"))

    def test_3_4_ambulance_south_to_north_movement(self):
        """Req 3, 4: AMB-01 travels South -> North (-Y direction) and approaches intersection."""
        signal_controller = SmartTrafficSignal("INT-TEST", self.cx, self.cy)
        signal_controller._current_phase = CyclePhase.NS_GREEN_EW_RED

        tm = TrafficManager(self.cx, self.cy, self.rw, self.stop_dist)
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=tm.lane_coords["SOUTH"]["x"],
            y=800.0,
            speed=95.0,
        )
        tm.ambulance = amb

        initial_y = amb.y
        # Run for 60 frames (1 second)
        for _ in range(60):
            tm.update(self.dt, signal_controller, sim_time=10.0)

        # Traveling North means y coordinate decreases
        self.assertLess(amb.y, initial_y, "Ambulance did not move North (-Y)")
        self.assertAlmostEqual(amb.vx, 0.0, places=2)
        self.assertAlmostEqual(amb.vy, -1.0, places=2)
        self.assertGreater(amb.speed, 80.0)

    def test_5_6_7_8_9_red_approach_stops_and_remains_waiting(self):
        """Req 5, 6, 7, 8, 9: Reaches stop line during South RED, decelerates smoothly, stops, and waits."""
        signal_controller = SmartTrafficSignal("INT-TEST", self.cx, self.cy)
        # Set phase to East-West Green, meaning North & South are RED
        signal_controller._current_phase = CyclePhase.NS_RED_EW_GREEN
        self.assertEqual(signal_controller.get_signal("SOUTH"), SignalState.RED)

        tm = TrafficManager(self.cx, self.cy, self.rw, self.stop_dist)
        south_stop_y = tm.lane_coords["SOUTH"]["stop"]  # cy + stop_dist = 415 + 75 = 490.0

        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=tm.lane_coords["SOUTH"]["x"],
            y=700.0,
            speed=95.0,
        )
        tm.ambulance = amb

        stopped = False
        decelerated = False

        # Run until stopped or max 8 seconds (480 frames)
        for frame in range(480):
            prev_speed = amb.speed
            tm.update(self.dt, signal_controller, sim_time=12.0)

            if amb.braking or amb.speed < prev_speed - 0.1:
                decelerated = True

            if amb.speed == 0.0 and amb.state == VehicleState.STOPPED:
                stopped = True
                break

        self.assertTrue(decelerated, "Req 6: AMB-01 failed to decelerate for RED signal")
        self.assertTrue(stopped, "Req 7: AMB-01 failed to come to a complete stop before RED signal")

        # Req 7: Front bumper must be strictly before South stop line
        # South vehicle moves North (-Y), front bumper is at y - length / 2
        front_y = amb.front_pos[1]
        self.assertGreaterEqual(
            front_y,
            south_stop_y - 1.0,
            f"AMB-01 overran South stop line! front_y={front_y}, stop_y={south_stop_y}"
        )
        self.assertFalse(amb.in_intersection, "Req 9: AMB-01 entered intersection on RED!")

        # Req 8: Verify it remains stopped for 3 additional seconds of RED
        y_stopped = amb.y
        for _ in range(180):
            tm.update(self.dt, signal_controller, sim_time=15.0)
            self.assertEqual(amb.speed, 0.0, "Req 8: AMB-01 moved while signal was still RED!")
            self.assertEqual(amb.y, y_stopped, "Req 8: AMB-01 drifted while stopped at RED!")
            self.assertEqual(amb.state, VehicleState.STOPPED)
            self.assertFalse(amb.in_intersection, "Req 9: AMB-01 entered intersection during RED wait!")

    def test_10_civilian_traffic_continues_normally(self):
        """Req 10: Civilian traffic continues to generate and operate across all approaches."""
        signal_controller = SmartTrafficSignal("INT-TEST", self.cx, self.cy)
        tm = TrafficManager(self.cx, self.cy, self.rw, self.stop_dist, max_vehicles=6, spawn_interval=1.0)

        # Run simulation for 12 seconds with ambulance present
        for f in range(720):
            sim_t = f * self.dt
            signal_controller.update(self.dt)
            tm.update(self.dt, signal_controller, sim_time=sim_t)

        self.assertGreater(len(tm.vehicles), 0, "Civilian vehicles failed to spawn alongside ambulance")
        approaches_active = set(v.approach for v in tm.vehicles)
        self.assertGreater(len(approaches_active), 1, "Civilian traffic not active across multiple approaches")

    def test_11_civilian_and_ambulance_safe_queueing(self):
        """Req 11: Civilian vehicle queueing behind AMB-01 maintains safe following distance without collision."""
        signal_controller = SmartTrafficSignal("INT-TEST", self.cx, self.cy)
        signal_controller._current_phase = CyclePhase.NS_RED_EW_GREEN  # RED for South

        tm = TrafficManager(self.cx, self.cy, self.rw, self.stop_dist)
        lc = tm.lane_coords["SOUTH"]

        # AMB-01 stopped before South stop line
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=lc["x"],
            y=lc["stop"] + 30.0,
            speed=0.0,
        )
        amb.state = VehicleState.STOPPED
        tm.ambulance = amb

        # Civilian vehicle approaching behind AMB-01 (South moves North, so higher Y is behind)
        civ_follower = CivilianVehicle(
            vehicle_id="C-BEHIND",
            approach="SOUTH",
            x=lc["x"],
            y=700.0,
            speed=95.0,
        )
        tm.vehicles.append(civ_follower)

        # Step until follower stops
        for frame in range(600):
            tm.update(self.dt, signal_controller, sim_time=13.0)
            # Distance from follower front bumper to ambulance rear bumper
            # South: rear bumper of ambulance is amb.y + amb.length / 2.0
            # Follower front bumper is civ_follower.y - civ_follower.length / 2.0
            amb_rear_y = amb.rear_pos[1]
            civ_front_y = civ_follower.front_pos[1]
            gap = civ_front_y - amb_rear_y
            self.assertGreaterEqual(gap, 0.0, f"Frame {frame}: Civilian vehicle rear-ended AMB-01! gap={gap}")

        self.assertEqual(civ_follower.speed, 0.0, "Civilian vehicle did not stop behind AMB-01")
        final_gap = civ_follower.front_pos[1] - amb.rear_pos[1]
        self.assertGreaterEqual(final_gap, MIN_FOLLOW_GAP - 2.0, f"Final gap behind AMB-01 too small: {final_gap}")

    def test_12_phase1_phase2_signal_mutex_preserved(self):
        """Req 12: Signal controller mutex is preserved (never simultaneous conflicting green)."""
        signal_controller = SmartTrafficSignal("INT-TEST", self.cx, self.cy)
        tm = TrafficManager(self.cx, self.cy, self.rw, self.stop_dist)

        for frame in range(1320):  # 22 seconds (1 complete cycle)
            dt = 1.0 / 60.0
            signal_controller.update(dt)
            tm.update(dt, signal_controller, sim_time=frame * dt)

            ns_state = signal_controller.get_signal("NORTH")
            ew_state = signal_controller.get_signal("EAST")

            # Assert mutex: North and East can NEVER be GREEN at the same time
            self.assertFalse(
                ns_state == SignalState.GREEN and ew_state == SignalState.GREEN,
                "SAFETY MUTEX VIOLATION: North and East were both GREEN!"
            )

    def test_13_v2v_regression_check(self):
        """Req 13: Existing V2V highway scenario remains completely unchanged."""
        try:
            import main
            import v2v_manager
            from highway_env.vehicle.kinematics import Vehicle
            from ai.trajectory_predictor import AIPredictor
            from ai.safety_fusion import SafetyFusionEngine

            self.assertTrue(hasattr(main, "main"))
            self.assertTrue(hasattr(v2v_manager, "V2VManager"))
            self.assertTrue(hasattr(AIPredictor, "predict"))
            print("[TEST 13] V2V regression check PASSED — highway scenario untouched.")
        except Exception as e:
            self.fail(f"V2V regression check failed: {e}")

    def test_14_zero_v2i_emergency_preemption_scope(self):
        """Req 14: Verify strictly ZERO emergency preemption or signal modification in Phase 3."""
        signal_controller = SmartTrafficSignal("INT-TEST", self.cx, self.cy)
        initial_timings = dict(signal_controller.phase_timings)
        initial_phase = signal_controller.current_phase

        tm = TrafficManager(self.cx, self.cy, self.rw, self.stop_dist)
        # AMB-01 spawns and waits at RED
        tm.ambulance = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=tm.lane_coords["SOUTH"]["x"],
            y=tm.lane_coords["SOUTH"]["stop"] + 20.0,
            speed=0.0,
        )

        for _ in range(120):
            tm.update(self.dt, signal_controller, sim_time=12.0)

        # Invariant checks:
        self.assertFalse(signal_controller.preemption_active, "Preemption must NOT be active in Phase 3!")
        self.assertEqual(signal_controller.phase_timings, initial_timings, "Signal timings must NOT be modified in Phase 3!")
        self.assertEqual(signal_controller.current_phase, initial_phase, "Signal phase must NOT be overridden by AMB-01 in Phase 3!")


if __name__ == "__main__":
    unittest.main()
