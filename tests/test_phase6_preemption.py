"""Phase 6: Comprehensive Verification Test Suite — Safe Emergency Signal Preemption.

Verifies:
  1. No emergency -> normal coordinated cycle runs without preemption.
  2. Valid emergency request initiates preemption transition sequence.
  3. No instantaneous RED -> GREEN: conflicting EW does not jump immediately to South green.
  4. Yellow clearance interval: conflicting EW transitions to PREEMPTION_YELLOW (3.0s).
  5. All-red clearance interval: transitions to PREEMPTION_ALL_RED where all 4 signals are RED.
  6. Safety gate blocks on conflict: vehicle in conflict zone holds ALL-RED.
  7. Safety gate clears enables green: once conflict clears, transitions to EMERGENCY_SOUTH_GREEN.
  8. Emergency South green: South is GREEN, East/West are RED.
  9. Mutual exclusion invariant: North/South and East/West never simultaneously GREEN.
 10. Normal cycle recovery: after 8.0s emergency green, transitions through yellow and recovers to normal cycle.
 11. Stale request rejected: outdated timestamp does not trigger preemption.
 12. Invalid request rejected: missing or malformed fields do not trigger preemption.
 13. Civilian traffic compliance: civilian vehicles obey preemption signals.
 14. V2I communication intact: range, latency, buffering work correctly.
 15. Continuous conflict analysis: analyzer runs throughout preemption sequence.
"""

import unittest

from v2i.smart_signal import SmartTrafficSignal, CyclePhase, SignalState
from v2i.traffic_analyzer import RSUTrafficAnalyzer, VehicleConflictStatus
from v2i.v2i_manager import V2IManager
from v2i.traffic_manager import TrafficManager, CivilianVehicle, AmbulanceVehicle


class TestPhase6SafePreemption(unittest.TestCase):
    """Phase 6 Safe Emergency Signal Preemption Verification."""

    def setUp(self):
        self.signal = SmartTrafficSignal(intersection_id="INT-01")
        self.v2i_channel = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=0.0)
        self.analyzer = self.signal.traffic_analyzer

    def _make_valid_request(self, current_time=10.0, approach="SOUTH"):
        return {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "position": (592.5, 550.0),
            "velocity": (0.0, -95.0),
            "approach": approach,
            "speed": 0.0,
            "timestamp": current_time,
            "priority": "HIGH",
            "eta": 0.0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: No Emergency -> Normal Cycle
    # ─────────────────────────────────────────────────────────────────────────
    def test_01_no_emergency_normal_cycle(self):
        """In the absence of an emergency request, signal runs normal coordinated cycle."""
        self.assertFalse(self.signal.preemption_requested)
        self.assertFalse(self.signal.preemption_active)
        self.assertFalse(self.signal.preemption_clearing)

        # Run through a full cycle
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_GREEN_EW_RED)
        self.signal.update(8.0)
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_YELLOW_EW_RED)
        self.signal.update(3.0)
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_RED_EW_GREEN)
        self.signal.update(8.0)
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_RED_EW_YELLOW)
        self.signal.update(3.0)
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_GREEN_EW_RED)

        # Preemption must never have activated
        self.assertFalse(self.signal.preemption_active)
        self.assertFalse(self.signal.preemption_requested)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: Emergency Request Triggers Preemption
    # ─────────────────────────────────────────────────────────────────────────
    def test_02_emergency_request_triggers_preemption(self):
        """Valid emergency request initiates preemption."""
        req = self._make_valid_request(current_time=5.0)
        accepted = self.signal.receive_emergency_request(req, current_time=5.0, auto_preempt=True)
        self.assertTrue(accepted)
        self.assertTrue(self.signal.preemption_requested)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: No Instantaneous RED -> GREEN Transitions
    # ─────────────────────────────────────────────────────────────────────────
    def test_03_no_instantaneous_green(self):
        """When EW is GREEN, South RED does not switch instantly to GREEN on request."""
        self.signal.current_phase = CyclePhase.NS_RED_EW_GREEN
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.RED)
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.GREEN)

        req = self._make_valid_request(current_time=5.0)
        self.signal.receive_emergency_request(req, current_time=5.0, auto_preempt=True)

        # Must NOT instantly become EMERGENCY_SOUTH_GREEN
        self.assertNotEqual(self.signal.current_phase, CyclePhase.EMERGENCY_SOUTH_GREEN)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.RED)
        self.assertEqual(self.signal.current_phase, CyclePhase.PREEMPTION_YELLOW)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: Yellow Clearance Interval
    # ─────────────────────────────────────────────────────────────────────────
    def test_04_yellow_clearance_interval(self):
        """Conflicting EW green transitions first to PREEMPTION_YELLOW for 3.0s."""
        self.signal.current_phase = CyclePhase.NS_RED_EW_GREEN
        self.signal.request_preemption("SOUTH")

        self.assertEqual(self.signal.current_phase, CyclePhase.PREEMPTION_YELLOW)
        self.assertTrue(self.signal.preemption_clearing)
        self.assertFalse(self.signal.preemption_active)
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.YELLOW)
        self.assertEqual(self.signal.get_signal("WEST"), SignalState.YELLOW)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.RED)
        self.assertEqual(self.signal.get_signal("NORTH"), SignalState.RED)

        # Advance 2.9s -> still in yellow clearance
        self.signal.update(2.9)
        self.assertEqual(self.signal.current_phase, CyclePhase.PREEMPTION_YELLOW)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 5: All-Red Clearance Interval
    # ─────────────────────────────────────────────────────────────────────────
    def test_05_all_red_clearance_interval(self):
        """After yellow clearance, signal transitions to PREEMPTION_ALL_RED (1.0s)."""
        self.signal.current_phase = CyclePhase.NS_RED_EW_GREEN
        self.signal.request_preemption("SOUTH")

        # Advance 3.0s through yellow
        self.signal.update(3.0)
        self.assertEqual(self.signal.current_phase, CyclePhase.PREEMPTION_ALL_RED)

        # All 4 signals MUST be strictly RED
        for app in ("NORTH", "SOUTH", "EAST", "WEST"):
            self.assertEqual(self.signal.get_signal(app), SignalState.RED)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 6: Safety Gate Blocks on Conflict
    # ─────────────────────────────────────────────────────────────────────────
    def test_06_safety_gate_blocks_on_conflict(self):
        """Vehicle inside the conflict zone holds signal in PREEMPTION_ALL_RED."""
        self.signal.current_phase = CyclePhase.PREEMPTION_ALL_RED
        self.signal.preemption_requested = True
        self.signal._time_in_phase = 0.0

        # Create a civilian vehicle in the central conflict zone
        v_in_zone = CivilianVehicle(
            vehicle_id="C01",
            approach="EAST",
            x=560.0,
            y=415.0,
            speed=20.0,
        )
        self.signal.update_traffic_analysis(civilian_vehicles=[v_in_zone], ambulance=None)

        # Advance 1.5s (> 1.0s duration)
        self.signal.update(1.5)

        # Must still be held in PREEMPTION_ALL_RED because conflict zone is occupied!
        self.assertEqual(self.signal.current_phase, CyclePhase.PREEMPTION_ALL_RED)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.RED)
        self.assertFalse(self.signal.preemption_active)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 7: Safety Gate Clears Enables Green
    # ─────────────────────────────────────────────────────────────────────────
    def test_07_safety_gate_clears_enables_green(self):
        """Once conflict zone clears, safety gate grants EMERGENCY_SOUTH_GREEN."""
        self.signal.current_phase = CyclePhase.PREEMPTION_ALL_RED
        self.signal.preemption_requested = True
        self.signal._time_in_phase = 0.0

        # Occupied initially
        v_clearing = CivilianVehicle(
            vehicle_id="C01",
            approach="EAST",
            x=560.0,
            y=415.0,
            speed=20.0,
        )
        self.signal.update_traffic_analysis(civilian_vehicles=[v_clearing], ambulance=None)
        self.signal.update(1.0)
        self.assertEqual(self.signal.current_phase, CyclePhase.PREEMPTION_ALL_RED)

        # Vehicle moves past the intersection (exited)
        v_clearing.x = 200.0  # Past West exit
        self.signal.update_traffic_analysis(civilian_vehicles=[v_clearing], ambulance=None)

        # Advance one step -> safety gate detects clear!
        self.signal.update(0.1)
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_SOUTH_GREEN)
        self.assertTrue(self.signal.preemption_active)
        self.assertFalse(self.signal.preemption_clearing)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.GREEN)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 8: Emergency South Green States
    # ─────────────────────────────────────────────────────────────────────────
    def test_08_emergency_south_green_states(self):
        """In EMERGENCY_SOUTH_GREEN: South is GREEN, East and West are RED."""
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.GREEN)
        self.assertEqual(self.signal.get_signal("NORTH"), SignalState.GREEN)
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.RED)
        self.assertEqual(self.signal.get_signal("WEST"), SignalState.RED)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 9: Mutual Exclusion Safety Invariant
    # ─────────────────────────────────────────────────────────────────────────
    def test_09_mutual_exclusion_invariant(self):
        """Assert North/South and East/West are NEVER simultaneously GREEN across all phases."""
        for phase in CyclePhase:
            self.signal.current_phase = phase
            ns_green = (
                self.signal.get_signal("NORTH") == SignalState.GREEN
                or self.signal.get_signal("SOUTH") == SignalState.GREEN
            )
            ew_green = (
                self.signal.get_signal("EAST") == SignalState.GREEN
                or self.signal.get_signal("WEST") == SignalState.GREEN
            )
            self.assertFalse(
                ns_green and ew_green,
                f"SAFETY VIOLATION in phase {phase}: Both NS and EW are GREEN!",
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Test 10: Normal Cycle Recovery
    # ─────────────────────────────────────────────────────────────────────────
    def test_10_normal_cycle_recovery(self):
        """After 8.0s of emergency green, transitions to yellow (3.0s) and recovers normal cycle."""
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        self.signal.preemption_active = True
        self.signal.preemption_requested = True

        # Advance 8.0s -> Emergency green expires
        self.signal.update(8.0)
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_YELLOW_EW_RED)
        self.assertFalse(self.signal.preemption_active)
        self.assertFalse(self.signal.preemption_requested)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.YELLOW)
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.RED)

        # Advance 3.0s -> Yellow clearance finishes, recovers to NS_RED_EW_GREEN
        self.signal.update(3.0)
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_RED_EW_GREEN)
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.GREEN)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.RED)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 11: Stale Request Rejected
    # ─────────────────────────────────────────────────────────────────────────
    def test_11_stale_request_rejected(self):
        """Request older than validity window (5.0s) is rejected and does not trigger preemption."""
        stale_req = self._make_valid_request(current_time=1.0)
        current_sim_time = 10.0  # Age = 9.0s > 5.0s

        accepted = self.signal.receive_emergency_request(
            stale_req,
            current_time=current_sim_time,
            auto_preempt=True,
        )
        self.assertFalse(accepted)
        self.assertFalse(self.signal.preemption_requested)
        self.assertFalse(self.signal.preemption_active)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 12: Invalid Request Rejected
    # ─────────────────────────────────────────────────────────────────────────
    def test_12_invalid_request_rejected(self):
        """Missing or malformed request fields do not trigger preemption."""
        invalid_req = {
            "type": "EMERGENCY_REQUEST",
            # missing vehicleId, approach, timestamp, etc.
        }
        accepted = self.signal.receive_emergency_request(
            invalid_req,
            current_time=5.0,
            auto_preempt=True,
        )
        self.assertFalse(accepted)
        self.assertFalse(self.signal.preemption_requested)
        self.assertFalse(self.signal.preemption_active)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 13: Civilian Traffic Compliance
    # ─────────────────────────────────────────────────────────────────────────
    def test_13_civilian_traffic_compliance(self):
        """Civilian vehicles obey signals during preemption phases."""
        tm = TrafficManager()
        tm.reset()

        # Place an East civilian vehicle approaching the stop line
        v_east = CivilianVehicle(
            vehicle_id="C_EW",
            approach="EAST",
            x=680.0,
            y=385.0,
            speed=40.0,
            target_speed=40.0,
        )
        tm.vehicles = [v_east]

        # Put signal into PREEMPTION_YELLOW
        self.signal.current_phase = CyclePhase.PREEMPTION_YELLOW
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.YELLOW)

        # Step traffic
        tm.update(dt=0.5, signal_controller=self.signal, sim_time=1.0)
        # Vehicle should be decelerating / braking for yellow
        self.assertTrue(v_east.braking or v_east.speed < 40.0)

        # Put signal into PREEMPTION_ALL_RED
        self.signal.current_phase = CyclePhase.PREEMPTION_ALL_RED
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.RED)

        # Step traffic until stopped
        for _ in range(50):
            tm.update(dt=0.1, signal_controller=self.signal, sim_time=2.0)

        # Must be stopped before stop line
        self.assertEqual(v_east.speed, 0.0)
        stop_x = tm.lane_coords["EAST"]["stop"]
        self.assertGreaterEqual(v_east.x, stop_x - 5.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 14: V2I Communication Link Intact
    # ─────────────────────────────────────────────────────────────────────────
    def test_14_v2i_communication_intact(self):
        """V2I channel transmits, buffers with latency, and delivers request correctly."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.5,
            y=550.0,
            speed=0.0,
        )
        rsu_pos = (560.0, 415.0)

        # Step communication at t=1.0s
        tx, state = amb.step_v2i_communication(
            1.0,
            rsu_pos,
            self.v2i_channel,
            515.0,
        )
        self.assertTrue(tx)
        self.assertEqual(state, "TRANSMITTED")

        # At t=1.1s (latency=0.2s), packet should still be in-flight
        delivered = self.v2i_channel.deliver_to_rsu(1.1)
        self.assertEqual(len(delivered), 0)

        # At t=1.25s, packet delivered!
        delivered = self.v2i_channel.deliver_to_rsu(1.25)
        self.assertEqual(len(delivered), 1)

        # Deliver to RSU controller
        accepted = self.signal.receive_emergency_request(
            delivered[0],
            current_time=1.25,
            auto_preempt=True,
        )
        self.assertTrue(accepted)
        self.assertTrue(self.signal.preemption_requested)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 15: Continuous Conflict Analysis During Preemption
    # ─────────────────────────────────────────────────────────────────────────
    def test_15_continuous_conflict_analysis_during_preemption(self):
        """RSUTrafficAnalyzer performs continuous monitoring throughout preemption."""
        req = self._make_valid_request(current_time=1.0)
        self.signal.receive_emergency_request(req, current_time=1.0, auto_preempt=True)

        v1 = CivilianVehicle(vehicle_id="C01", approach="WEST", x=300.0, y=445.0, speed=30.0)
        v2 = CivilianVehicle(vehicle_id="C02", approach="EAST", x=700.0, y=385.0, speed=35.0)
        vehicles = [v1, v2]

        analysis = self.signal.update_traffic_analysis(vehicles, ambulance=None, current_time=1.5)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.emergency_vehicle_id, "AMB-01")
        self.assertEqual(analysis.conflicting_approaches, ["EAST", "WEST"])
        self.assertIsNotNone(self.signal.latest_analysis)


if __name__ == "__main__":
    unittest.main()
