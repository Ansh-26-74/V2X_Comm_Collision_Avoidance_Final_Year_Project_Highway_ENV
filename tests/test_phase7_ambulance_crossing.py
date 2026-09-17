"""Phase 7: Comprehensive Verification Test Suite — Emergency Vehicle Safe Intersection Crossing.

Verifies:
  1. Ambulance remains stopped at RED before emergency authorization.
  2. Ambulance becomes authorized when South signal becomes GREEN.
  3. Ambulance accelerates smoothly from stop position without impossible acceleration.
  4. Stop line crossing: AMB-01 physically crosses the South stop line.
  5. Intersection entry: AMB-01 enters the central intersection box.
  6. Crossing state: while physically inside intersection, state is CROSSING_INTERSECTION.
  7. Correct direction: AMB-01 travels strictly South -> North without heading drift.
  8. No conflicting GREEN: East/West signals remain strictly RED during crossing.
  9. Collision safety: zero bounding-box overlaps with civilian vehicles throughout crossing.
 10. Civilian traffic safety: conflicting civilian traffic remains stopped at red signals.
 11. No stopping inside intersection: AMB-01 never halts inside junction box.
 12. Exit detection: AMB-01 is marked cleared ONLY when full body (rear bumper) passes northern exit.
 13. Cleared event: exactly one logical 'AMB-01 CLEARED INTERSECTION' event occurs.
 14. Telemetry: telemetry reflects live dynamic states and speeds.
 15. Phases 4/5/6 regression: V2I comm, conflict analysis, and preemption remain fully functional.
"""

import unittest
import math

from v2i.smart_signal import SmartTrafficSignal, CyclePhase, SignalState
from v2i.traffic_manager import (
    TrafficManager,
    CivilianVehicle,
    AmbulanceVehicle,
    VehicleState,
    AmbulanceState,
)
from v2i.v2i_manager import V2IManager


class TestPhase7AmbulanceCrossing(unittest.TestCase):
    """Phase 7 Emergency Vehicle Safe Intersection Crossing Verification."""

    def setUp(self):
        self.signal = SmartTrafficSignal(intersection_id="INT-01")
        self.tm = TrafficManager(
            center_x=560,
            center_y=415,
            road_width=130,
            stop_line_dist=75,
            max_vehicles=6,
            spawn_interval=3.0,
            ambulance_spawn_time=0.0,
        )
        self.dt = 1.0 / 60.0
        self.south_stop_y = self.tm.lane_coords["SOUTH"]["stop"]     # 490.0
        self.south_enter_y = self.tm.lane_coords["SOUTH"]["enter"]   # 480.0
        self.north_exit_y = self.tm.lane_coords["SOUTH"]["exit"]     # 350.0

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: Ambulance Remains Stopped at RED
    # ─────────────────────────────────────────────────────────────────────────
    def test_01_ambulance_remains_stopped_at_red(self):
        """Before emergency South GREEN, AMB-01 remains stopped safely behind stop line."""
        self.signal.current_phase = CyclePhase.NS_RED_EW_GREEN
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.RED)

        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=self.tm.lane_coords["SOUTH"]["x"],
            y=600.0,
            speed=95.0,
        )
        self.tm.ambulance = amb
        self.tm._ambulance_spawned = True

        # Run until AMB-01 comes to a full stop before RED
        stopped = False
        for _ in range(300):
            self.tm.update(self.dt, self.signal, sim_time=1.0)
            if amb.speed == 0.0:
                stopped = True
                break

        self.assertTrue(stopped, "AMB-01 failed to come to a complete stop before RED signal")
        self.assertFalse(amb.past_stop_line, "AMB-01 crossed stop line on RED!")
        self.assertFalse(amb.in_intersection, "AMB-01 entered intersection on RED!")
        self.assertFalse(amb.is_authorized, "AMB-01 should NOT be authorized while RED")
        self.assertGreaterEqual(amb.front_pos[1], self.south_stop_y - 1.0)

        # Confirm it stays stopped for another 120 frames (2 seconds)
        y_before = amb.y
        for _ in range(120):
            self.tm.update(self.dt, self.signal, sim_time=3.0)
            self.assertEqual(amb.speed, 0.0)
            self.assertEqual(amb.y, y_before)
            self.assertFalse(amb.past_stop_line)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: Ambulance Authorized on South GREEN
    # ─────────────────────────────────────────────────────────────────────────
    def test_02_ambulance_authorized_on_south_green(self):
        """When South signal becomes GREEN, AMB-01 becomes authorized to proceed."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=self.tm.lane_coords["SOUTH"]["x"],
            y=545.0,
            speed=0.0,
        )
        self.tm.ambulance = amb
        self.tm._ambulance_spawned = True
        self.assertFalse(amb.is_authorized)

        # Signal grants South Green
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.GREEN)

        self.tm.update(self.dt, self.signal, sim_time=10.0)
        self.assertTrue(amb.is_authorized, "AMB-01 failed to become authorized on South GREEN")
        self.assertIn(
            amb.state,
            (VehicleState.AUTHORIZED_TO_PROCEED, VehicleState.ACCELERATING),
            "AMB-01 state should reflect authorization or acceleration",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Ambulance Accelerates Smoothly
    # ─────────────────────────────────────────────────────────────────────────
    def test_03_ambulance_accelerates_smoothly(self):
        """After authorization, AMB-01 accelerates smoothly from 0 with bounded acceleration."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=self.tm.lane_coords["SOUTH"]["x"],
            y=545.0,
            speed=0.0,
        )
        self.tm.ambulance = amb
        self.tm._ambulance_spawned = True
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        prev_speed = 0.0
        accelerated = False
        for frame in range(60):  # 1 second
            self.tm.update(self.dt, self.signal, sim_time=10.0 + frame * self.dt)
            curr_speed = amb.speed
            accel = (curr_speed - prev_speed) / self.dt
            # Verify no impossible acceleration or reverse speed
            self.assertGreaterEqual(curr_speed, 0.0, "Negative speed detected!")
            self.assertLessEqual(accel, amb.max_emergency_accel + 5.0, "Impossible acceleration detected!")
            if curr_speed > 10.0:
                accelerated = True
            prev_speed = curr_speed

        self.assertTrue(accelerated, "AMB-01 failed to accelerate smoothly after authorization")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: Stop Line Crossing
    # ─────────────────────────────────────────────────────────────────────────
    def test_04_stop_line_crossing(self):
        """AMB-01 physically crosses the South stop line."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=self.tm.lane_coords["SOUTH"]["x"],
            y=545.0,
            speed=0.0,
        )
        self.tm.ambulance = amb
        self.tm._ambulance_spawned = True
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        crossed = False
        for frame in range(120):
            self.tm.update(self.dt, self.signal, sim_time=10.0 + frame * self.dt)
            if amb.past_stop_line:
                crossed = True
                break

        self.assertTrue(crossed, "AMB-01 failed to cross South stop line")
        self.assertLessEqual(amb.front_pos[1], self.south_stop_y)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 5: Intersection Entry
    # ─────────────────────────────────────────────────────────────────────────
    def test_05_intersection_entry(self):
        """AMB-01 enters the central intersection."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=self.tm.lane_coords["SOUTH"]["x"],
            y=545.0,
            speed=0.0,
        )
        self.tm.ambulance = amb
        self.tm._ambulance_spawned = True
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        entered = False
        for frame in range(180):
            self.tm.update(self.dt, self.signal, sim_time=10.0 + frame * self.dt)
            if amb.in_intersection:
                entered = True
                break

        self.assertTrue(entered, "AMB-01 failed to enter intersection")
        self.assertLessEqual(amb.front_pos[1], self.south_enter_y)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 6: Crossing State
    # ─────────────────────────────────────────────────────────────────────────
    def test_06_crossing_state(self):
        """While physically inside intersection, state is CROSSING_INTERSECTION."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=self.tm.lane_coords["SOUTH"]["x"],
            y=415.0,  # Exactly center of intersection
            speed=60.0,
        )
        amb.is_authorized = True
        amb.past_stop_line = True
        amb.in_intersection = True
        self.tm.ambulance = amb
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        self.tm.update(self.dt, self.signal, sim_time=12.0)
        self.assertEqual(amb.state, VehicleState.CROSSING_INTERSECTION)
        self.assertEqual(amb.ambulance_state, AmbulanceState.CROSSING_INTERSECTION)
        self.assertEqual(amb.state.value, "CROSSING_INTERSECTION")
        self.assertEqual(amb.state, "CROSSING_INTERSECTION")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 7: Correct Direction
    # ─────────────────────────────────────────────────────────────────────────
    def test_07_correct_direction(self):
        """AMB-01 continues South -> North without heading drift."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=self.tm.lane_coords["SOUTH"]["x"],
            y=545.0,
            speed=0.0,
        )
        self.tm.ambulance = amb
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        initial_x = amb.x
        for frame in range(180):
            self.tm.update(self.dt, self.signal, sim_time=10.0 + frame * self.dt)
            self.assertAlmostEqual(amb.x, initial_x, delta=0.5, msg="Ambulance drifted laterally!")
            self.assertAlmostEqual(amb.heading, -math.pi / 2.0, places=3, msg="Ambulance heading drifted!")
            self.assertEqual(amb.vx, 0.0)
            self.assertEqual(amb.vy, -1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 8: No Conflicting GREEN
    # ─────────────────────────────────────────────────────────────────────────
    def test_08_no_conflicting_green(self):
        """During ambulance crossing, East and West signals remain strictly RED."""
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.RED)
        self.assertEqual(self.signal.get_signal("WEST"), SignalState.RED)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.GREEN)

        # Advance 7.9s of emergency green
        for _ in range(79):
            self.signal.update(0.1)
            self.assertEqual(self.signal.get_signal("EAST"), SignalState.RED)
            self.assertEqual(self.signal.get_signal("WEST"), SignalState.RED)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 9: No Collision Throughout Crossing
    # ─────────────────────────────────────────────────────────────────────────
    def test_09_no_collision(self):
        """Zero bounding-box overlaps between AMB-01 and civilian vehicles."""
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=self.tm.lane_coords["SOUTH"]["x"],
            y=545.0,
            speed=0.0,
        )
        self.tm.ambulance = amb

        # Add civilian vehicles on East, West, and North approaches
        v_east = CivilianVehicle(vehicle_id="C_E", approach="EAST", x=700.0, y=382.5, speed=0.0)
        v_west = CivilianVehicle(vehicle_id="C_W", approach="WEST", x=400.0, y=447.5, speed=0.0)
        v_north = CivilianVehicle(vehicle_id="C_N", approach="NORTH", x=527.5, y=300.0, speed=0.0)
        self.tm.vehicles = [v_east, v_west, v_north]

        for frame in range(240):  # 4 seconds
            self.tm.update(self.dt, self.signal, sim_time=10.0 + frame * self.dt)
            for civ in self.tm.vehicles:
                collision = amb.collides_with(civ)
                self.assertFalse(collision, f"Collision detected between AMB-01 and {civ.vehicle_id} at frame {frame}!")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 10: Civilian Traffic Safety Compliance
    # ─────────────────────────────────────────────────────────────────────────
    def test_10_civilian_traffic_safety(self):
        """Conflicting East/West civilian traffic remains stopped at red stop lines."""
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        v_east = CivilianVehicle(
            vehicle_id="C_E",
            approach="EAST",
            x=680.0,
            y=382.5,
            speed=30.0,
            target_speed=30.0,
        )
        self.tm.vehicles = [v_east]

        for _ in range(120):
            self.tm.update(self.dt, self.signal, sim_time=10.0)

        # East vehicle must be stopped before East stop line
        self.assertEqual(v_east.speed, 0.0)
        self.assertFalse(v_east.in_intersection)
        self.assertGreaterEqual(v_east.x, self.tm.lane_coords["EAST"]["stop"] - 5.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 11: No Stopping Inside Intersection
    # ─────────────────────────────────────────────────────────────────────────
    def test_11_no_stopping_inside_intersection(self):
        """Once AMB-01 crosses stop line, signal changes do not halt it inside intersection."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=self.tm.lane_coords["SOUTH"]["x"],
            y=440.0,  # Inside junction box
            speed=50.0,
        )
        amb.is_authorized = True
        amb.past_stop_line = True
        amb.in_intersection = True
        self.tm.ambulance = amb

        # Force signal to RED to test that in-box vehicle is never stopped
        self.signal.current_phase = CyclePhase.NS_YELLOW_EW_RED

        for _ in range(60):
            self.tm.update(self.dt, self.signal, sim_time=15.0)
            self.assertGreater(amb.speed, 0.0, "AMB-01 stopped inside intersection on signal change!")
            self.assertFalse(amb.braking, "AMB-01 braked inside intersection!")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 12: Exit Detection by Full Vehicle Body
    # ─────────────────────────────────────────────────────────────────────────
    def test_12_exit_detection_full_body(self):
        """AMB-01 is marked cleared ONLY when its full body (rear bumper) passes northern exit."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=self.tm.lane_coords["SOUTH"]["x"],
            y=365.0,  # Front bumper is 365 - 22 = 343 (passed 350 exit), but rear is 365 + 22 = 387 (inside!)
            speed=40.0,
        )
        amb.is_authorized = True
        amb.past_stop_line = True
        amb.in_intersection = True
        self.tm.ambulance = amb
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        # Step one frame: front is past exit, but rear is still inside
        self.tm.update(0.01, self.signal, sim_time=12.0)
        self.assertFalse(amb.cleared, "Marked cleared prematurely before rear body cleared exit!")
        self.assertEqual(amb.state, VehicleState.EXITING_INTERSECTION)

        # Move forward until rear passes 350.0 (y < 350 - 22 = 328.0)
        amb.y = 320.0  # Rear is 320 + 22 = 342 <= 350 -> fully cleared!
        self.tm.update(0.01, self.signal, sim_time=12.1)
        self.assertTrue(amb.cleared, "Failed to mark cleared after full body passed northern exit")
        self.assertEqual(amb.state, VehicleState.CLEARED)
        self.assertFalse(amb.in_intersection)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 13: Cleared Event Occurs Exactly Once
    # ─────────────────────────────────────────────────────────────────────────
    def test_13_cleared_event_occurs_once(self):
        """Verify exactly one logical 'AMB-01 CLEARED INTERSECTION' event occurs."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=self.tm.lane_coords["SOUTH"]["x"],
            y=350.0,
            speed=60.0,
        )
        amb.is_authorized = True
        amb.past_stop_line = True
        amb.in_intersection = True
        self.tm.ambulance = amb
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        # Advance through exit
        for _ in range(60):
            self.tm.update(self.dt, self.signal, sim_time=13.0)

        self.assertTrue(amb.cleared)
        self.assertEqual(
            len(self.tm.ambulance_cleared_events),
            1,
            f"Expected exactly 1 cleared event, got {len(self.tm.ambulance_cleared_events)}",
        )
        self.assertEqual(self.tm.ambulance_cleared_events[0], "AMB-01 CLEARED INTERSECTION")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 14: Telemetry Reflects Actual Live States
    # ─────────────────────────────────────────────────────────────────────────
    def test_14_telemetry_reflects_actual_state(self):
        """Verify ambulance exposes live state and speed without hardcoded values."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=self.tm.lane_coords["SOUTH"]["x"],
            y=415.0,
            speed=72.0,
        )
        amb.is_authorized = True
        amb.past_stop_line = True
        amb.in_intersection = True
        self.tm.ambulance = amb
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        self.tm.update(self.dt, self.signal, sim_time=14.0)
        kmh = amb.speed * 0.36
        self.assertGreater(kmh, 25.0)
        self.assertLess(kmh, 45.0)
        self.assertEqual(amb.state, VehicleState.CROSSING_INTERSECTION)
        self.assertEqual(amb.ambulance_state, AmbulanceState.CROSSING_INTERSECTION)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 15: Phases 4, 5, 6 Regression
    # ─────────────────────────────────────────────────────────────────────────
    def test_15_phase_4_5_6_regression(self):
        """V2I comm, traffic analysis, and safe preemption sequence continue operating."""
        v2i = V2IManager(v2i_range=400.0, latency=0.2)
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.5,
            y=550.0,
            speed=0.0,
        )
        rsu_pos = (560.0, 415.0)

        # Step V2I communication
        tx, status = amb.step_v2i_communication(1.0, rsu_pos, v2i, 515.0)
        self.assertTrue(tx)
        delivered = v2i.deliver_to_rsu(1.25)
        self.assertEqual(len(delivered), 1)

        # RSU receives and runs safe preemption
        accepted = self.signal.receive_emergency_request(delivered[0], current_time=1.25, auto_preempt=True)
        self.assertTrue(accepted)
        self.assertTrue(self.signal.preemption_requested)

        # Conflict analysis
        analysis = self.signal.update_traffic_analysis(civilian_vehicles=[], ambulance=amb, current_time=1.25)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.emergency_vehicle_id, "AMB-01")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 16: Deterministic Overtaking of Blocked Civilian on South GREEN
    # ─────────────────────────────────────────────────────────────────────────
    def test_16_deterministic_overtaking_of_blocked_civilian(self):
        """When South signal is GREEN and a civilian vehicle is stopped ahead,
        AMB-01 must perform a smooth, collision-free lane change into the adjacent
        same-direction lane (x=628), pass the vehicle, cross the stop line,
        traverse the intersection, and exit North with exactly one cleared event.
        """
        # 1. Setup civilian vehicle stopped ahead at South stop line
        civ = CivilianVehicle(
            vehicle_id="C01",
            approach="SOUTH",
            x=592.0,
            y=514.0,  # Stopped ahead before stop line at 490.0
            speed=0.0,
            target_speed=0.0,
        )
        self.tm.vehicles.append(civ)

        # 2. Setup AMB-01 stopped safely behind C01
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=560.0,  # 46px behind civ
            speed=0.0,
            target_speed=110.0,
        )
        self.tm.ambulance = amb
        self.tm._ambulance_spawned = True

        # Initially South is RED: verify AMB-01 waits behind civilian without overlap
        self.signal.current_phase = CyclePhase.NS_RED_EW_GREEN
        self.tm.update(self.dt, self.signal, sim_time=1.0)
        self.assertFalse(amb.collides_with(civ))
        self.assertEqual(amb.speed, 0.0)

        # 3. Trigger South GREEN preemption
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.GREEN)

        # Track history for invariants
        positions_x = []
        positions_y = []
        states = []
        had_overtaking_state = False
        collided = False

        # Run simulation for 300 frames (~5.0 seconds)
        for frame in range(300):
            t = 2.0 + frame * self.dt
            self.tm.update(self.dt, self.signal, sim_time=t)

            positions_x.append(amb.x)
            positions_y.append(amb.y)
            states.append(amb.ambulance_state)

            # Continuous collision invariant (Section 13)
            if amb.collides_with(civ):
                collided = True

            if amb.ambulance_state == AmbulanceState.OVERTAKING or amb.state == VehicleState.OVERTAKING:
                had_overtaking_state = True

            if len(positions_x) >= 2:
                dx = abs(positions_x[-1] - positions_x[-2])
                # Smooth lateral movement: max step at 45 px/s * dt = ~0.75 px
                self.assertLessEqual(
                    dx,
                    1.5,
                    f"Frame {frame}: Teleportation detected! Lateral step was {dx:.2f}px",
                )

            # Direction must remain South -> North (y non-increasing)
            if len(positions_y) >= 2:
                self.assertLessEqual(
                    positions_y[-1],
                    positions_y[-2] + 0.001,
                    f"Frame {frame}: Reverse movement detected! y went from {positions_y[-2]} to {positions_y[-1]}",
                )

            if amb.cleared:
                break

        # Verifications
        # 1 & 2: Never collided or overlapped
        self.assertFalse(collided, "AMB-01 overlapped civilian vehicle during overtaking!")

        # 3, 4, 5, 6: Overtaking occurred
        self.assertTrue(had_overtaking_state, "AMB-01 never entered OVERTAKING state!")

        # 8: Reached adjacent lane (x = 628.0)
        self.assertAlmostEqual(amb.x, 628.0, delta=1.0, msg="AMB-01 did not settle into adjacent lane x=628.0")

        # 9: Passed civilian vehicle (amb.y < civ.y)
        self.assertLess(amb.y, civ.y, "AMB-01 did not pass the blocking civilian vehicle!")

        # 10, 11, 12, 13: Crossed stop line and cleared intersection
        self.assertTrue(amb.past_stop_line, "AMB-01 did not cross the stop line!")
        self.assertTrue(amb.cleared, "AMB-01 did not clear the intersection!")
        self.assertLessEqual(amb.rear_pos[1], self.north_exit_y + 0.1, "AMB-01 full body did not clear northern exit!")

        # 16: Exactly one cleared event emitted
        self.assertEqual(len(self.tm.ambulance_cleared_events), 1)
        self.assertEqual(self.tm.ambulance_cleared_events[0], "AMB-01 CLEARED INTERSECTION")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 17: Negative Safety — Target Lane Occupied
    # ─────────────────────────────────────────────────────────────────────────
    def test_17_no_overtake_when_target_lane_occupied(self):
        """If adjacent same-direction lane (x=628) is blocked by another civilian,
        AMB-01 must NOT overtake, must NOT collide, and must hold safely behind C01.
        """
        # C01 blocking Lane 1
        civ1 = CivilianVehicle(
            vehicle_id="C01",
            approach="SOUTH",
            x=592.0,
            y=515.0,
            speed=0.0,
            target_speed=0.0,
        )
        # C02 blocking Lane 2 abreast
        civ2 = CivilianVehicle(
            vehicle_id="C02",
            approach="SOUTH",
            x=628.0,
            y=520.0,
            speed=0.0,
            target_speed=0.0,
        )
        self.tm.vehicles.extend([civ1, civ2])

        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=560.0,
            speed=0.0,
            target_speed=110.0,
        )
        self.tm.ambulance = amb
        self.tm._ambulance_spawned = True
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        # Step 120 frames (~2.0 seconds)
        for frame in range(120):
            t = 1.0 + frame * self.dt
            self.tm.update(self.dt, self.signal, sim_time=t)

            self.assertFalse(amb.collides_with(civ1), "Collided with C01!")
            self.assertFalse(amb.collides_with(civ2), "Collided with C02!")

        # AMB-01 must remain in Lane 1 behind C01 without changing lanes
        self.assertAlmostEqual(amb.x, 592.0, delta=1.0)
        self.assertFalse(amb.is_overtaking)
        self.assertGreater(amb.front_pos[1], civ1.rear_pos[1])

    # ─────────────────────────────────────────────────────────────────────────
    # Test 18: Negative Safety — Vehicle Approaching Fast in Target Lane
    # ─────────────────────────────────────────────────────────────────────────
    def test_18_no_overtake_when_target_lane_vehicle_approaching(self):
        """If a vehicle is approaching fast from behind in the target lane,
        AMB-01 must not cut in front of it.
        """
        civ1 = CivilianVehicle(
            vehicle_id="C01",
            approach="SOUTH",
            x=592.0,
            y=515.0,
            speed=0.0,
        )
        # Fast vehicle approaching from behind in Lane 2 (y=610, speed=100)
        civ2 = CivilianVehicle(
            vehicle_id="C02",
            approach="SOUTH",
            x=628.0,
            y=600.0,  # 40px behind amb (y=560)
            speed=100.0,
            target_speed=100.0,
        )
        self.tm.vehicles.extend([civ1, civ2])

        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=560.0,
            speed=0.0,
        )
        self.tm.ambulance = amb
        self.tm._ambulance_spawned = True
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        # Verify evaluate_overtaking rejects the lane change
        can_ot = amb.evaluate_overtaking(self.tm.vehicles, self.south_stop_y, self.south_enter_y)
        self.assertFalse(can_ot, "evaluate_overtaking should reject lane change with fast approaching vehicle!")

        # Step 30 frames: AMB-01 should not enter Lane 2
        for frame in range(30):
            self.tm.update(self.dt, self.signal, sim_time=1.0 + frame * self.dt)
            self.assertFalse(amb.collides_with(civ1))
            self.assertFalse(amb.collides_with(civ2))

        self.assertAlmostEqual(amb.x, 592.0, delta=1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 19: Negative Safety — No Overtaking Inside Intersection Box
    # ─────────────────────────────────────────────────────────────────────────
    def test_19_no_overtake_inside_intersection(self):
        """AMB-01 must NEVER initiate a new overtaking maneuver inside the central
        intersection conflict zone (Section 9 constraint).
        """
        # AMB-01 is inside intersection (y = 420.0, enter_y = 480.0)
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=420.0,
            speed=80.0,
        )
        amb.is_authorized = True
        amb.past_stop_line = True
        amb.in_intersection = True

        civ = CivilianVehicle(
            vehicle_id="C01",
            approach="SOUTH",
            x=592.0,
            y=390.0,
            speed=30.0,
        )

        can_ot = amb.evaluate_overtaking([civ], self.south_stop_y, self.south_enter_y)
        self.assertFalse(can_ot, "Overtaking inside intersection box must be strictly rejected!")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 20: Negative Safety — No Blocking Vehicle (Normal Phase 7 Flow)
    # ─────────────────────────────────────────────────────────────────────────
    def test_20_no_overtake_when_lane_is_clear(self):
        """When South lane is clear, AMB-01 proceeds normally in Lane 1 (x=592)
        without initiating an unnecessary lane change.
        """
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=560.0,
            speed=0.0,
            target_speed=110.0,
        )
        self.tm.ambulance = amb
        self.tm._ambulance_spawned = True
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        for frame in range(250):
            t = 1.0 + frame * self.dt
            self.tm.update(self.dt, self.signal, sim_time=t)
            # Must remain in Lane 1
            self.assertAlmostEqual(amb.x, 592.0, delta=0.5)
            self.assertFalse(amb.is_overtaking)
            if amb.cleared:
                break

        self.assertTrue(amb.cleared)
        self.assertAlmostEqual(amb.x, 592.0, delta=0.5)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 21: Negative Safety — Multiple Queued Vehicles Hold Safely
    # ─────────────────────────────────────────────────────────────────────────
    def test_21_multiple_queued_vehicles_hold_safely(self):
        """Verify AMB-01 holds safely behind multiple queued civilian vehicles
        when both lanes are occupied, with zero collisions.
        """
        civ1 = CivilianVehicle(vehicle_id="C01", approach="SOUTH", x=592.0, y=500.0, speed=0.0, target_speed=0.0)
        civ2 = CivilianVehicle(vehicle_id="C02", approach="SOUTH", x=592.0, y=535.0, speed=0.0, target_speed=0.0)
        civ3 = CivilianVehicle(vehicle_id="C03", approach="SOUTH", x=628.0, y=510.0, speed=0.0, target_speed=0.0)
        self.tm.vehicles.extend([civ1, civ2, civ3])

        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=580.0,
            speed=0.0,
        )
        self.tm.ambulance = amb
        self.tm._ambulance_spawned = True
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        for frame in range(100):
            self.tm.update(self.dt, self.signal, sim_time=1.0 + frame * self.dt)
            for v in [civ1, civ2, civ3]:
                self.assertFalse(amb.collides_with(v))

        self.assertEqual(amb.speed, 0.0)
        self.assertFalse(amb.is_overtaking)


if __name__ == "__main__":
    unittest.main()

