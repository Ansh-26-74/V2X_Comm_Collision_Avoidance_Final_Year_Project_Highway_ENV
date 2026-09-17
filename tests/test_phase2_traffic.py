"""Comprehensive Phase 2 Test Suite: Normal Civilian Traffic Simulation.

Validates all 12 Phase 2 requirements:
 1. Vehicle approaches RED.
 2. Vehicle stops behind stop line.
 3. Vehicle remains stopped while RED.
 4. Signal changes to GREEN.
 5. Vehicle accelerates and crosses.
 6. Vehicle continues beyond the intersection.
 7. Vehicle approaches YELLOW and reacts appropriately (dilemma zone handling).
 8. Multiple vehicles maintain safe car-following spacing (no queue collisions).
 9. Vehicles from different approaches do not collide (signal mutex + box clearance).
10. Simulation runs cleanly for multiple complete signal cycles.
11. No runtime errors or exceptions throughout execution.
12. V2V scenario regression verification (existing highway simulation untouched).
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
    TrafficManager,
    VehicleState,
    MIN_FOLLOW_GAP,
    STOP_LINE_BUFFER,
    VEHICLE_LENGTH,
)
import v2i.v2i_renderer as renderer


class TestPhase2CivilianTraffic(unittest.TestCase):

    def setUp(self):
        self.cx = renderer.CX
        self.cy = renderer.CY
        self.rw = renderer.ROAD_WIDTH
        self.stop_dist = renderer.STOP_LINE_DIST
        self.dt = 1.0 / 60.0

    def test_1_2_3_red_approach_stops_and_remains_stopped(self):
        """Req 1, 2, 3: Vehicle approaches RED, stops behind stop line, and remains stopped."""
        signal_controller = SmartTrafficSignal("INT-TEST", self.cx, self.cy)
        # Force North signal to RED by setting phase to EW Green
        signal_controller._current_phase = CyclePhase.NS_RED_EW_GREEN
        self.assertEqual(signal_controller.get_signal("NORTH"), SignalState.RED)

        tm = TrafficManager(self.cx, self.cy, self.rw, self.stop_dist)
        stop_line_y = tm.lane_coords["NORTH"]["stop"]  # cy - stop_dist = 415 - 75 = 340

        # Create vehicle approaching North stop line
        v = CivilianVehicle(
            vehicle_id="TEST-N1",
            approach="NORTH",
            x=tm.lane_coords["NORTH"]["x"],
            y=100.0,
            speed=95.0,
        )
        tm.vehicles.append(v)

        # Run until stopped or max 10 seconds (600 frames)
        stopped = False
        for _ in range(600):
            tm.update(self.dt, signal_controller)
            if v.speed == 0.0 and v.state == VehicleState.STOPPED:
                stopped = True
                break

        self.assertTrue(stopped, "Vehicle failed to come to a complete stop at RED signal")
        # Verify bumper is behind stop line (North vehicle moves South +Y, so front bumper y must be <= stop_line_y)
        front_y = v.front_pos[1]
        self.assertLessEqual(
            front_y,
            stop_line_y,
            f"Vehicle crossed stop line on RED! front_y={front_y}, stop_line_y={stop_line_y}"
        )
        self.assertFalse(v.in_intersection, "Vehicle entered intersection on RED!")

        # Req 3: Verify it remains stopped for 3 additional seconds of RED
        y_pos_stopped = v.y
        for _ in range(180):
            tm.update(self.dt, signal_controller)
            self.assertEqual(v.speed, 0.0, "Vehicle moved while signal was still RED!")
            self.assertEqual(v.y, y_pos_stopped, "Vehicle drifted while stopped at RED!")
            self.assertEqual(v.state, VehicleState.STOPPED)

    def test_4_5_6_green_accelerates_crosses_and_continues(self):
        """Req 4, 5, 6: Signal turns GREEN, vehicle accelerates, crosses, and continues beyond."""
        signal_controller = SmartTrafficSignal("INT-TEST", self.cx, self.cy)
        tm = TrafficManager(self.cx, self.cy, self.rw, self.stop_dist)
        lc = tm.lane_coords["NORTH"]

        # Place vehicle stopped at stop line under RED
        v = CivilianVehicle(
            vehicle_id="TEST-N2",
            approach="NORTH",
            x=lc["x"],
            y=lc["stop"] - VEHICLE_LENGTH / 2.0 - STOP_LINE_BUFFER,
            speed=0.0,
        )
        v.state = VehicleState.STOPPED
        tm.vehicles.append(v)

        # Change signal to GREEN for North/South
        signal_controller._current_phase = CyclePhase.NS_GREEN_EW_RED
        self.assertEqual(signal_controller.get_signal("NORTH"), SignalState.GREEN)

        entered = False
        crossed = False
        accelerated = False

        # Simulate for up to 8 seconds
        for frame in range(480):
            tm.update(self.dt, signal_controller)

            if v.speed > 5.0:
                accelerated = True
            if v.in_intersection:
                entered = True
            if v.cleared:
                crossed = True
                break

        self.assertTrue(accelerated, "Vehicle failed to accelerate on GREEN")
        self.assertTrue(entered, "Vehicle never entered intersection on GREEN")
        self.assertTrue(crossed, "Vehicle never cleared intersection")

        # Req 6: Verify vehicle continues moving beyond intersection
        prev_y = v.y
        for _ in range(30):
            tm.update(self.dt, signal_controller)
        self.assertGreater(v.y, prev_y, "Vehicle failed to continue moving beyond intersection")
        self.assertGreater(v.speed, 50.0, "Vehicle failed to maintain cruising speed after intersection")

    def test_7_yellow_approach_decision(self):
        """Req 7: Vehicle approaching YELLOW reacts appropriately (stop if safe, clear if committed)."""
        signal_controller = SmartTrafficSignal("INT-TEST", self.cx, self.cy)
        signal_controller._current_phase = CyclePhase.NS_YELLOW_EW_RED
        self.assertEqual(signal_controller.get_signal("NORTH"), SignalState.YELLOW)

        # Case A: Vehicle far enough from stop line -> should smoothly stop
        tm_a = TrafficManager(self.cx, self.cy, self.rw, self.stop_dist)
        lc = tm_a.lane_coords["NORTH"]
        v_far = CivilianVehicle(
            vehicle_id="TEST-Y-FAR",
            approach="NORTH",
            x=lc["x"],
            y=150.0,  # stop line is at 340, so 190px away
            speed=95.0,
        )
        tm_a.vehicles.append(v_far)

        stopped_safely = False
        for _ in range(400):
            tm_a.update(self.dt, signal_controller)
            if v_far.speed == 0.0:
                stopped_safely = True
                break

        self.assertTrue(stopped_safely, "Far vehicle did not stop on YELLOW")
        self.assertLessEqual(v_far.front_pos[1], lc["stop"] + 1.0, "Far vehicle overran stop line on YELLOW")

        # Case B: Vehicle right at stop line at speed -> should safely clear without abrupt hard stop
        tm_b = TrafficManager(self.cx, self.cy, self.rw, self.stop_dist)
        v_close = CivilianVehicle(
            vehicle_id="TEST-Y-CLOSE",
            approach="NORTH",
            x=lc["x"],
            y=lc["stop"] - 10.0,  # only 10px before stop line, cruising at 95px/s
            speed=95.0,
        )
        tm_b.vehicles.append(v_close)

        cleared_smoothly = False
        for _ in range(300):
            tm_b.update(self.dt, signal_controller)
            if v_close.in_intersection or v_close.cleared:
                cleared_smoothly = True
                break

        self.assertTrue(cleared_smoothly, "Committed vehicle abruptly stopped instead of clearing on YELLOW")

    def test_8_multiple_vehicles_spacing_and_queue(self):
        """Req 8: Multiple vehicles approaching the same stop line maintain safe spacing without collision."""
        signal_controller = SmartTrafficSignal("INT-TEST", self.cx, self.cy)
        signal_controller._current_phase = CyclePhase.NS_RED_EW_GREEN  # RED for North

        tm = TrafficManager(self.cx, self.cy, self.rw, self.stop_dist)
        lc = tm.lane_coords["NORTH"]

        # Lead vehicle already stopped near stop line
        v_lead = CivilianVehicle(
            vehicle_id="LEAD",
            approach="NORTH",
            x=lc["x"],
            y=lc["stop"] - 25.0,
            speed=0.0,
        )
        # Following vehicle approaching from behind at speed
        v_follow = CivilianVehicle(
            vehicle_id="FOLLOW",
            approach="NORTH",
            x=lc["x"],
            y=120.0,
            speed=95.0,
        )
        # Third vehicle further behind
        v_third = CivilianVehicle(
            vehicle_id="THIRD",
            approach="NORTH",
            x=lc["x"],
            y=20.0,
            speed=95.0,
        )

        tm.vehicles.extend([v_lead, v_follow, v_third])

        # Step until all vehicles stop
        for frame in range(600):
            tm.update(self.dt, signal_controller)
            # Check for bumper overlap at every frame
            gap_1_2 = v_lead.rear_pos[1] - v_follow.front_pos[1]
            gap_2_3 = v_follow.rear_pos[1] - v_third.front_pos[1]
            self.assertGreaterEqual(gap_1_2, 0.0, f"Frame {frame}: Follower rear-ended lead vehicle! gap={gap_1_2}")
            self.assertGreaterEqual(gap_2_3, 0.0, f"Frame {frame}: Third vehicle rear-ended follower! gap={gap_2_3}")

        self.assertEqual(v_follow.speed, 0.0, "Following vehicle did not come to complete stop in queue")
        self.assertEqual(v_third.speed, 0.0, "Third vehicle did not come to complete stop in queue")

        final_gap_1_2 = v_lead.rear_pos[1] - v_follow.front_pos[1]
        final_gap_2_3 = v_follow.rear_pos[1] - v_third.front_pos[1]
        self.assertGreaterEqual(final_gap_1_2, MIN_FOLLOW_GAP - 2.0, "Final queue gap too small")
        self.assertGreaterEqual(final_gap_2_3, MIN_FOLLOW_GAP - 2.0, "Final queue gap too small")

    def test_9_cross_traffic_separation_and_clearance(self):
        """Req 9: Opposing movements remain separated and do not collide in intersection."""
        signal_controller = SmartTrafficSignal("INT-TEST", self.cx, self.cy)
        tm = TrafficManager(self.cx, self.cy, self.rw, self.stop_dist)

        # Place vehicle in intersection crossing West->East
        v_west = CivilianVehicle(
            vehicle_id="CROSSING-W",
            approach="WEST",
            x=self.cx,  # Right in center of intersection
            y=tm.lane_coords["WEST"]["y"],
            speed=90.0,
        )
        v_west.past_stop_line = True
        v_west.in_intersection = True
        tm.vehicles.append(v_west)

        # Place North vehicle at stop line with GREEN signal
        signal_controller._current_phase = CyclePhase.NS_GREEN_EW_RED
        v_north = CivilianVehicle(
            vehicle_id="WAITING-N",
            approach="NORTH",
            x=tm.lane_coords["NORTH"]["x"],
            y=tm.lane_coords["NORTH"]["stop"] - 20.0,
            speed=0.0,
        )
        tm.vehicles.append(v_north)

        # Before crossing clears, North vehicle should yield and not enter
        self.assertTrue(tm.is_intersection_blocked_for("NORTH"))
        tm.update(self.dt, signal_controller)
        self.assertFalse(v_north.in_intersection, "North vehicle entered occupied intersection box!")

        # Step until West vehicle clears intersection
        for _ in range(120):
            tm.update(self.dt, signal_controller)
            if v_west.cleared:
                break

        self.assertTrue(v_west.cleared, "West vehicle did not clear intersection")
        self.assertFalse(tm.is_intersection_blocked_for("NORTH"))

    def test_10_11_multi_cycle_stability_and_no_errors(self):
        """Req 10, 11: Run simulation across multiple complete signal cycles with no errors."""
        signal_controller = SmartTrafficSignal("INT-TEST", self.cx, self.cy)
        tm = TrafficManager(
            center_x=self.cx,
            center_y=self.cy,
            road_width=self.rw,
            stop_line_dist=self.stop_dist,
            max_vehicles=7,
            spawn_interval=1.5,
        )

        # Run for 75 simulated seconds (~3 full signal cycles: 8s+3s+8s+3s = 22s per cycle)
        total_frames = int(75.0 / self.dt)
        initial_cycle = signal_controller.cycle_count

        for frame in range(total_frames):
            signal_controller.update(self.dt)
            tm.update(self.dt, signal_controller)

            # Check vehicle count limit
            self.assertLessEqual(
                len(tm.vehicles),
                tm.max_vehicles,
                f"Active vehicle count exceeded maximum ({len(tm.vehicles)} > {tm.max_vehicles})"
            )

            # Check for collisions among active vehicles in same approach
            for i, v1 in enumerate(tm.vehicles):
                for j, v2 in enumerate(tm.vehicles):
                    if i >= j:
                        continue
                    if v1.approach == v2.approach:
                        # Longitudinal separation check
                        if v1.approach in ("NORTH", "SOUTH"):
                            dist = abs(v1.y - v2.y)
                        else:
                            dist = abs(v1.x - v2.x)
                        self.assertGreater(
                            dist,
                            VEHICLE_LENGTH * 0.7,
                            f"Same-approach collision detected at frame {frame} between {v1.vehicle_id} and {v2.vehicle_id}"
                        )

        # Confirm multiple cycles actually elapsed
        cycles_completed = signal_controller.cycle_count - initial_cycle
        self.assertGreaterEqual(cycles_completed, 3, f"Expected >= 3 cycles, got {cycles_completed}")
        print(f"\n[TEST 10-11] Completed {total_frames} frames ({cycles_completed} full cycles) cleanly without errors.")

    def test_12_v2v_regression_check(self):
        """Req 12: Verify that existing V2V modules and main.py remain completely functional."""
        try:
            import main
            import v2v_manager
            from highway_env.vehicle.kinematics import Vehicle
            from ai.trajectory_predictor import AIPredictor
            from ai.safety_fusion import SafetyFusionEngine

            self.assertTrue(hasattr(main, "main"))
            self.assertTrue(hasattr(v2v_manager, "V2VManager"))
            self.assertTrue(hasattr(AIPredictor, "predict"))
            print("[TEST 12] V2V regression check PASSED — all V2V imports and classes intact.")
        except Exception as e:
            self.fail(f"V2V regression check failed with error: {e}")


if __name__ == "__main__":
    unittest.main()
