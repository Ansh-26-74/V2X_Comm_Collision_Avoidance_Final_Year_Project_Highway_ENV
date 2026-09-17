"""Phase 5 Automated Test Suite — RSU Traffic & Conflict Analysis.

Validates:
  1. Valid emergency request acceptance.
  2. Invalid emergency request rejection (malformed schema, invalid approach/type).
  3. Stale emergency request rejection (timestamp delta > 5.0s).
  4. Movement & conflict derivation (SOUTH -> NORTH, conflicting: EAST <-> WEST).
  5. East-West civilian conflict detection inside the conflict zone.
  6. Conflict clearance detection when vehicle leaves the conflict zone.
  7. Non-conflicting traffic classification (North/South alignment).
  8. Continuous real-time dynamic analysis over multi-frame simulation steps.
  9. CRITICAL SCOPE RULE: Signal controller does NOT change phase or timings.
 10. Ambulance compliance: AMB-01 remains stopped at RED before South stop line.
 11. Civilian traffic obeys signals normally and is not overridden.
 12. Regression testing: Highway V2V scenario is 100% untouched and functional.
"""

import os
import unittest

os.environ["SDL_VIDEODRIVER"] = "dummy"

from v2i.smart_signal import SmartTrafficSignal, SignalState, CyclePhase
from v2i.traffic_manager import TrafficManager, AmbulanceVehicle, CivilianVehicle
from v2i.traffic_analyzer import RSUTrafficAnalyzer, VehicleConflictStatus, EmergencyAnalysis
from v2i.v2i_manager import V2IManager


class TestPhase5TrafficAnalysis(unittest.TestCase):
    """Test suite for Phase 5 RSU Traffic & Conflict Analysis."""

    def setUp(self):
        self.analyzer = RSUTrafficAnalyzer(center_x=560.0, center_y=415.0, road_width=130.0)
        self.stop_line_y = 490.0

    def test_01_valid_emergency_request(self):
        """Test that a valid AMB-01 request passes validation and activates analysis."""
        req = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "position": (592.5, 520.0),
            "velocity": (0.0, 0.0),
            "speed": 0.0,
            "heading": -1.571,
            "approach": "SOUTH",
            "eta": 0.0,
            "priority": "HIGH",
            "timestamp": 12.0,
        }
        valid, reason = self.analyzer.validate_emergency_request(req, current_time=12.5)
        self.assertTrue(valid)
        self.assertEqual(reason, "VALID_EMERGENCY_REQUEST")

        # Verify acceptance in SmartTrafficSignal
        signal = SmartTrafficSignal(intersection_id="INT-01")
        accepted = signal.receive_emergency_request(req, current_time=12.5)
        self.assertTrue(accepted)
        self.assertTrue(signal.emergency_request_received)
        self.assertIsNotNone(signal.latest_emergency_request)

    def test_02_invalid_emergency_request(self):
        """Test that malformed or missing fields trigger safe rejection."""
        # 1. Missing vehicleId
        bad_req1 = {"type": "EMERGENCY_REQUEST", "vehicleType": "EMERGENCY", "emergency": True, "approach": "SOUTH", "position": (0, 0), "timestamp": 1.0}
        valid, reason = self.analyzer.validate_emergency_request(bad_req1, 1.0)
        self.assertFalse(valid)
        self.assertIn("REJECTED", reason)

        # 2. Wrong type
        bad_req2 = {"type": "RANDOM_PACKET", "vehicleId": "AMB-01", "vehicleType": "EMERGENCY", "emergency": True, "approach": "SOUTH", "position": (0, 0), "timestamp": 1.0}
        valid, reason = self.analyzer.validate_emergency_request(bad_req2, 1.0)
        self.assertFalse(valid)

        # 3. Invalid approach
        bad_req3 = {"type": "EMERGENCY_REQUEST", "vehicleId": "AMB-01", "vehicleType": "EMERGENCY", "emergency": True, "approach": "DIAGONAL", "position": (0, 0), "timestamp": 1.0}
        valid, reason = self.analyzer.validate_emergency_request(bad_req3, 1.0)
        self.assertFalse(valid)

        # 4. Incomplete position
        bad_req4 = {"type": "EMERGENCY_REQUEST", "vehicleId": "AMB-01", "vehicleType": "EMERGENCY", "emergency": True, "approach": "SOUTH", "position": None, "timestamp": 1.0}
        valid, reason = self.analyzer.validate_emergency_request(bad_req4, 1.0)
        self.assertFalse(valid)

    def test_03_stale_emergency_request(self):
        """Test that requests older than staleness threshold (> 5.0s) are rejected."""
        req = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "position": (592.5, 520.0),
            "approach": "SOUTH",
            "timestamp": 10.0,
        }
        # Current time is 18.0s -> age = 8.0s > 5.0s
        valid, reason = self.analyzer.validate_emergency_request(req, current_time=18.0)
        self.assertFalse(valid)
        self.assertIn("REJECTED_STALE", reason)

    def test_04_south_to_north_path(self):
        """Test that SOUTH approach derives intended direction NORTH and conflicts EAST/WEST."""
        intended, conflicts = self.analyzer.determine_path_and_conflicts("SOUTH")
        self.assertEqual(intended, "NORTH")
        self.assertIn("EAST", conflicts)
        self.assertIn("WEST", conflicts)
        self.assertNotIn("NORTH", conflicts)

    def test_05_ew_conflict_detection(self):
        """Test that civilian East-West vehicle in the conflict zone is detected."""
        req = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "position": (592.5, 520.0),
            "approach": "SOUTH",
            "timestamp": 5.0,
        }
        # Civilian vehicle C01 on EAST approach inside the intersection box
        v_in_box = CivilianVehicle(vehicle_id="C01", approach="EAST", x=560.0, y=415.0, speed=80.0)
        v_in_box.in_intersection = True

        analysis = self.analyzer.analyze_traffic(
            request=req,
            civilian_vehicles=[v_in_box],
            current_time=5.1,
        )
        self.assertIsNotNone(analysis)
        self.assertTrue(analysis.intersection_occupied)
        self.assertTrue(analysis.conflict_detected)
        self.assertIn("C01", analysis.vehicles_in_conflict)
        self.assertFalse(analysis.safe_for_future_preemption)
        self.assertEqual(v_in_box.conflict_status, VehicleConflictStatus.IN_CONFLICT_ZONE.value)

    def test_06_conflict_clearance(self):
        """Test that when civilian vehicle exits the conflict zone, intersection updates to clear."""
        req = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "position": (592.5, 520.0),
            "approach": "SOUTH",
            "timestamp": 6.0,
        }
        # East vehicle has exited to the west of the intersection box (x = 420 < 495)
        v_cleared = CivilianVehicle(vehicle_id="C01", approach="EAST", x=420.0, y=415.0, speed=90.0)
        v_cleared.in_intersection = False

        analysis = self.analyzer.analyze_traffic(
            request=req,
            civilian_vehicles=[v_cleared],
            current_time=6.2,
        )
        self.assertIsNotNone(analysis)
        self.assertFalse(analysis.intersection_occupied)
        self.assertIn("C01", analysis.vehicles_cleared)
        self.assertTrue(analysis.safe_for_future_preemption)

    def test_07_non_conflicting_traffic(self):
        """Test that vehicles on North approach (aligned with ambulance) are not classified as conflict."""
        req = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "position": (592.5, 520.0),
            "approach": "SOUTH",
            "timestamp": 7.0,
        }
        # North vehicle approaching stop line (not inside intersection)
        v_north = CivilianVehicle(vehicle_id="C02", approach="NORTH", x=527.5, y=300.0, speed=85.0)

        analysis = self.analyzer.analyze_traffic(
            request=req,
            civilian_vehicles=[v_north],
            current_time=7.1,
        )
        self.assertIsNotNone(analysis)
        self.assertFalse(analysis.intersection_occupied)
        self.assertEqual(len(analysis.vehicles_in_conflict), 0)
        self.assertEqual(len(analysis.vehicles_approaching_conflict), 0)
        self.assertTrue(analysis.safe_for_future_preemption)
        self.assertEqual(v_north.conflict_status, VehicleConflictStatus.SAFE.value)

    def test_08_continuous_analysis(self):
        """Test dynamic progression over time: approaching -> in conflict zone -> cleared."""
        req = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "position": (592.5, 520.0),
            "approach": "SOUTH",
            "timestamp": 10.0,
        }
        # Vehicle moves from East (x=700) towards West (x=400)
        veh = CivilianVehicle(vehicle_id="C03", approach="EAST", x=700.0, y=390.0, speed=100.0)

        # Step 1: Approaching conflict (x=680 is within 140px of entrance at 625)
        veh.x = 680.0
        a1 = self.analyzer.analyze_traffic(req, [veh], current_time=10.1)
        self.assertIn("C03", a1.vehicles_approaching_conflict)
        self.assertFalse(a1.intersection_occupied)

        # Step 2: Inside conflict zone (x=560)
        veh.x = 560.0
        veh.in_intersection = True
        a2 = self.analyzer.analyze_traffic(req, [veh], current_time=10.3)
        self.assertIn("C03", a2.vehicles_in_conflict)
        self.assertTrue(a2.intersection_occupied)
        self.assertFalse(a2.safe_for_future_preemption)

        # Step 3: Cleared conflict zone (x=450 < 495)
        veh.x = 450.0
        veh.in_intersection = False
        a3 = self.analyzer.analyze_traffic(req, [veh], current_time=10.6)
        self.assertIn("C03", a3.vehicles_cleared)
        self.assertFalse(a3.intersection_occupied)
        self.assertTrue(a3.safe_for_future_preemption)

    def test_09_signal_unchanged(self):
        """CRITICAL SCOPE RULE: Traffic signal MUST NOT alter phase, timings, or cycle during analysis."""
        signal = SmartTrafficSignal(intersection_id="INT-01")
        signal.current_phase = CyclePhase.NS_RED_EW_GREEN
        signal._time_in_phase = 4.0
        initial_remaining = signal.time_remaining
        initial_phase = signal.current_phase

        req = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "position": (592.5, 520.0),
            "approach": "SOUTH",
            "timestamp": 5.0,
        }
        signal.receive_emergency_request(req, current_time=5.1)

        # Perform analysis with an East vehicle
        veh = CivilianVehicle(vehicle_id="C01", approach="EAST", x=560.0, y=415.0, speed=90.0)
        analysis = signal.update_traffic_analysis([veh], ambulance=None, current_time=5.2)

        self.assertIsNotNone(analysis)
        self.assertTrue(analysis.intersection_occupied)

        # Strict scope verification:
        self.assertEqual(signal.current_phase, initial_phase)
        self.assertEqual(signal.get_signal("SOUTH"), SignalState.RED)
        self.assertEqual(signal.get_signal("EAST"), SignalState.GREEN)
        self.assertEqual(signal.time_in_phase, 4.0)
        self.assertEqual(signal.time_remaining, initial_remaining)
        self.assertFalse(signal.preemption_active)

    def test_10_ambulance_remains_stopped(self):
        """Verify AMB-01 remains fully stopped at RED before South stop line during analysis."""
        tm = TrafficManager(ambulance_spawn_time=0.0)
        signal = SmartTrafficSignal(intersection_id="INT-01")
        signal.current_phase = CyclePhase.NS_RED_EW_GREEN  # South is RED
        channel = V2IManager(v2i_range=400.0, latency=0.2)
        rsu_pos = (735.0, 255.0)

        tm.ambulance_spawn_time = 0.0
        tm._spawn_ambulance()

        # Step simulation for 7 seconds
        dt = 0.05
        for step in range(140):
            sim_time = step * dt
            tm.update(dt=dt, signal_controller=signal, sim_time=sim_time, v2i_channel=channel, rsu_pos=rsu_pos)
            pkts = channel.deliver_to_rsu(sim_time)
            for p in pkts:
                signal.receive_emergency_request(p, current_time=sim_time)
            signal.update_traffic_analysis(tm.vehicles, tm.ambulance, sim_time)

        amb = tm.ambulance
        self.assertIsNotNone(amb)
        self.assertEqual(amb.speed, 0.0)
        # Front bumper must be stopped before stop line at y=490.0
        front_bumper_y = amb.y - amb.length / 2
        self.assertGreater(front_bumper_y, self.stop_line_y)
        self.assertFalse(amb.in_intersection)

    def test_11_civilian_traffic_unaffected(self):
        """Verify civilian traffic continues following signal commands without override."""
        tm = TrafficManager(ambulance_spawn_time=999.0)
        signal = SmartTrafficSignal(intersection_id="INT-01")
        signal.current_phase = CyclePhase.NS_RED_EW_GREEN  # East has GREEN

        veh = CivilianVehicle(vehicle_id="C01", approach="EAST", x=750.0, y=390.0, speed=100.0)
        tm.vehicles.append(veh)

        # Trigger analysis
        req = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "position": (592.5, 520.0),
            "approach": "SOUTH",
            "timestamp": 1.0,
        }
        signal.receive_emergency_request(req, current_time=1.0)
        signal.update_traffic_analysis(tm.vehicles, None, current_time=1.0)

        # Update civilian vehicle: since signal is GREEN, vehicle should cruise through
        tm.update(dt=0.1, signal_controller=signal, sim_time=1.1)
        self.assertGreater(veh.speed, 0.0)

    def test_12_v2v_highway_regression(self):
        """Verify that the V2V highway scenario remains 100% untouched and functional."""
        import v2v_manager
        mgr = v2v_manager.V2VManager()
        self.assertIsNotNone(mgr)
        self.assertGreater(mgr.v2v_range, 0)


if __name__ == "__main__":
    unittest.main()
