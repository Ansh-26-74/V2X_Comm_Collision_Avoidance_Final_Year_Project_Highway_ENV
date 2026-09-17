"""Phase 8 Test Suite: Emergency Priority Termination & Normal-Cycle Recovery.

Tests:
    1. Normal emergency termination and recovery sequence
       (EMERGENCY_SOUTH_GREEN -> EMERGENCY_TERMINATING -> RECOVERY_ALL_RED -> NS_RED_EW_GREEN)
    2. No instant green switch (never direct South GREEN -> EW GREEN or Normal GREEN)
    3. Strict mutual exclusion invariant throughout entire recovery sequence
    4. Duplicate clearance event safety (idempotent, no timer restart or state corruption)
    5. No premature emergency termination (before full-body clearance)
    6. Emergency request buffer retirement (consumed request cannot re-trigger preemption)
    7. Normal signal operation without emergency (Phase 8 remains inactive)
    8. Civilian traffic safety and compliance following recovery
    9. Ambulance continuous forward movement after clearance (no stop, reverse, teleport)
    10. RSU priority status lifecycle tracking (ACTIVE -> TERMINATING -> INACTIVE)
    11. Full end-to-end headless integration run through all phases
"""

import unittest
import pygame

from v2i.smart_signal import SmartTrafficSignal, CyclePhase, SignalState
from v2i.traffic_manager import (
    TrafficManager,
    CivilianVehicle,
    AmbulanceVehicle,
    VehicleState,
    AmbulanceState,
)
from v2i.v2i_manager import V2IManager


class TestPhase8TerminationRecovery(unittest.TestCase):
    """Test suite covering Phase 8 emergency termination and normal cycle recovery."""

    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        self.signal = SmartTrafficSignal(
            intersection_id="INT-01",
            intersection_x=560.0,
            intersection_y=415.0,
            phase_timings={
                CyclePhase.NS_GREEN_EW_RED: 8.0,
                CyclePhase.NS_YELLOW_EW_RED: 3.0,
                CyclePhase.NS_RED_EW_GREEN: 8.0,
                CyclePhase.NS_RED_EW_YELLOW: 3.0,
                CyclePhase.PREEMPTION_YELLOW: 3.0,
                CyclePhase.PREEMPTION_ALL_RED: 1.0,
                CyclePhase.EMERGENCY_SOUTH_GREEN: 8.0,
                CyclePhase.EMERGENCY_TERMINATING: 3.0,
                CyclePhase.RECOVERY_ALL_RED: 1.0,
            },
        )
        self.traffic = TrafficManager(
            center_x=560.0,
            center_y=415.0,
            road_width=130.0,
            stop_line_dist=65.0,
        )
        self.v2i_channel = V2IManager(v2i_range=400.0, latency=0.0, packet_loss_rate=0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: Full Normal Recovery Sequence
    # ─────────────────────────────────────────────────────────────────────────
    def test_01_normal_emergency_recovery_sequence(self):
        """Verify the complete controlled termination and recovery sequence:
        EMERGENCY_SOUTH_GREEN -> EMERGENCY_TERMINATING -> RECOVERY_ALL_RED -> NS_RED_EW_GREEN.
        """
        # Place controller into EMERGENCY_SOUTH_GREEN
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        self.signal.preemption_active = True
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.GREEN)
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.RED)
        self.assertEqual(self.signal.get_signal("WEST"), SignalState.RED)

        # Fire clearance event
        res = self.signal.notify_ambulance_cleared("AMB-01")
        self.assertTrue(res)

        # 1. State must immediately become EMERGENCY_TERMINATING
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_TERMINATING)
        self.assertEqual(self.signal.time_in_phase, 0.0)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.YELLOW)
        self.assertEqual(self.signal.get_signal("NORTH"), SignalState.YELLOW)
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.RED)
        self.assertEqual(self.signal.get_signal("WEST"), SignalState.RED)

        # Step through yellow clearance (3.0 seconds)
        self.signal.update(1.5)
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_TERMINATING)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.YELLOW)
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.RED)

        # Complete yellow duration
        self.signal.update(1.6)
        # 2. Must transition to RECOVERY_ALL_RED
        self.assertEqual(self.signal.current_phase, CyclePhase.RECOVERY_ALL_RED)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.RED)
        self.assertEqual(self.signal.get_signal("NORTH"), SignalState.RED)
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.RED)
        self.assertEqual(self.signal.get_signal("WEST"), SignalState.RED)

        # Step through all-red duration (1.0 second)
        self.signal.update(1.1)
        # 3. Must transition to normal cycle starting at NS_RED_EW_GREEN
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_RED_EW_GREEN)
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.GREEN)
        self.assertEqual(self.signal.get_signal("WEST"), SignalState.GREEN)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.RED)
        self.assertEqual(self.signal.get_signal("NORTH"), SignalState.RED)

        # 4. Continue normal cycle: EW GREEN -> EW YELLOW -> NS GREEN
        self.signal.update(8.1)
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_RED_EW_YELLOW)
        self.signal.update(3.1)
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_GREEN_EW_RED)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.GREEN)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: No Instant Green Switch
    # ─────────────────────────────────────────────────────────────────────────
    def test_02_no_instant_green_switch(self):
        """Immediately after clearance notification, verify signal does NOT switch to EW GREEN."""
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        self.signal.preemption_active = True

        self.signal.notify_ambulance_cleared("AMB-01")

        # Must not be any normal green phase
        self.assertNotEqual(self.signal.current_phase, CyclePhase.NS_RED_EW_GREEN)
        self.assertNotEqual(self.signal.current_phase, CyclePhase.NS_GREEN_EW_RED)
        self.assertNotEqual(self.signal.get_signal("EAST"), SignalState.GREEN)
        self.assertNotEqual(self.signal.get_signal("WEST"), SignalState.GREEN)

        # Must be controlled YELLOW
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_TERMINATING)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.YELLOW)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Mutual Exclusion During Whole Recovery
    # ─────────────────────────────────────────────────────────────────────────
    def test_03_mutual_exclusion_invariant_throughout_recovery(self):
        """Assert NS GREEN and EW GREEN can never both be true at any moment during recovery."""
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        self.signal.preemption_active = True
        self.signal.notify_ambulance_cleared("AMB-01")

        # Step at 60Hz dt for 15 seconds through yellow, all-red, EW green, EW yellow, NS green
        dt = 1.0 / 60.0
        for _ in range(900):
            self.signal.update(dt)
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
                f"MUTEX VIOLATION at phase {self.signal.current_phase.value}: NS={ns_green}, EW={ew_green}",
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: Duplicate Clear Event Safety (Idempotence)
    # ─────────────────────────────────────────────────────────────────────────
    def test_04_duplicate_clear_event_safety(self):
        """Duplicate clearance calls must be safely ignored without restarting timers or corrupting states."""
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        self.signal.preemption_active = True

        first_res = self.signal.notify_ambulance_cleared("AMB-01")
        self.assertTrue(first_res)
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_TERMINATING)

        # Advance 1.5 seconds into 3.0s yellow
        self.signal.update(1.5)
        self.assertAlmostEqual(self.signal.time_in_phase, 1.5, places=2)

        # Inject second identical clear event
        second_res = self.signal.notify_ambulance_cleared("AMB-01")
        self.assertFalse(second_res)

        # Timer must NOT have reset to 0.0
        self.assertAlmostEqual(self.signal.time_in_phase, 1.5, places=2)
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_TERMINATING)

        # Inject third clear event while in all-red
        self.signal.update(1.6)
        self.assertEqual(self.signal.current_phase, CyclePhase.RECOVERY_ALL_RED)
        third_res = self.signal.notify_ambulance_cleared("AMB-01")
        self.assertFalse(third_res)
        self.assertEqual(self.signal.current_phase, CyclePhase.RECOVERY_ALL_RED)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 5: No Premature Clearance Trigger
    # ─────────────────────────────────────────────────────────────────────────
    def test_05_no_premature_clearance(self):
        """Ensure AMB-01 crossing stop line or inside intersection does NOT trigger clearance."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=520.0,  # Approaching South stop line (stop line at y=480.0)
            speed=90.0,
        )
        self.traffic.ambulance = amb
        lc = self.traffic.lane_coords["SOUTH"]

        # Step while approaching
        amb.update_control(
            dt=0.1,
            signal_state=SignalState.GREEN,
            stop_line_coord=lc["stop"],
            intersection_enter=lc["enter"],
            intersection_exit=lc["exit"],
            lead_vehicle=None,
        )
        self.assertFalse(amb.cleared)
        self.assertFalse(amb.cleared_event_fired)
        self.assertEqual(len(self.traffic.ambulance_cleared_events), 0)

        # Cross stop line into intersection (y=415.0)
        amb.y = 415.0
        amb.update_control(
            dt=0.1,
            signal_state=SignalState.GREEN,
            stop_line_coord=lc["stop"],
            intersection_enter=lc["enter"],
            intersection_exit=lc["exit"],
            lead_vehicle=None,
        )
        self.assertTrue(amb.in_intersection)
        self.assertFalse(amb.cleared)
        self.assertFalse(amb.cleared_event_fired)
        self.assertEqual(len(self.traffic.ambulance_cleared_events), 0)

        # Front bumper exits northern boundary (exit at y=350.0) but rear bumper still inside
        # amb length is 44.0, front_pos=(x, y - 22.0), rear_pos=(x, y + 22.0)
        amb.y = 360.0  # front=338 (past exit), rear=382 (inside intersection)
        amb.update_control(
            dt=0.1,
            signal_state=SignalState.GREEN,
            stop_line_coord=lc["stop"],
            intersection_enter=lc["enter"],
            intersection_exit=lc["exit"],
            lead_vehicle=None,
        )
        self.assertFalse(amb.cleared)
        self.assertFalse(amb.cleared_event_fired)
        self.assertEqual(len(self.traffic.ambulance_cleared_events), 0)

        # Entire rear bumper clears exit boundary (rear <= 350.0, so y <= 328.0)
        amb.y = 320.0  # rear=342 <= 350
        amb.update_control(
            dt=0.1,
            signal_state=SignalState.GREEN,
            stop_line_coord=lc["stop"],
            intersection_enter=lc["enter"],
            intersection_exit=lc["exit"],
            lead_vehicle=None,
        )
        self.assertTrue(amb.cleared)
        self.assertTrue(amb.cleared_event_fired)

        # When traffic manager update runs, it consumes the cleared event and notifies signal controller
        self.traffic.update(
            dt=0.01,
            signal_controller=self.signal,
            sim_time=1.0,
        )
        self.assertEqual(len(self.traffic.ambulance_cleared_events), 1)
        self.assertEqual(self.traffic.ambulance_cleared_events[0], "AMB-01 CLEARED INTERSECTION")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 6: Request Buffer Retirement
    # ─────────────────────────────────────────────────────────────────────────
    def test_06_emergency_request_buffer_retirement(self):
        """Ensure consumed emergency request buffer cannot re-trigger preemption."""
        req_msg = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "approach": "SOUTH",
            "priority": "HIGH",
            "speed": 85.0,
            "eta": 2.5,
            "timestamp": 10.0,
            "position": (592.0, 600.0),
        }
        # Receive and preempt
        accepted = self.signal.receive_emergency_request(req_msg, current_time=10.0, auto_preempt=True)
        self.assertTrue(accepted)
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN

        # Ambulance clears
        self.signal.notify_ambulance_cleared("AMB-01")
        self.assertTrue(self.signal.latest_emergency_request["consumed"])
        self.assertFalse(self.signal.latest_emergency_request["active"])
        self.assertTrue(self.signal.emergency_completed)

        # Step through recovery to normal operation
        self.signal.update(3.1)  # yellow -> all-red
        self.signal.update(1.1)  # all-red -> NS_RED_EW_GREEN
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_RED_EW_GREEN)

        # Try sending another request from same completed AMB-01
        res = self.signal.receive_emergency_request(req_msg, current_time=15.0, auto_preempt=True)
        self.assertFalse(res)
        # Signal must remain in normal coordinated cycle, not re-preempted
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_RED_EW_GREEN)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 7: Normal Operation Without Emergency
    # ─────────────────────────────────────────────────────────────────────────
    def test_07_normal_operation_without_emergency(self):
        """When no emergency occurs, Phase 8 code remains inactive and cycle runs normally."""
        self.assertEqual(self.signal.rsu_priority_status, "INACTIVE")
        self.assertFalse(self.signal.emergency_terminating)
        self.assertFalse(self.signal.emergency_completed)

        phases_visited = []
        for _ in range(600):  # 10 seconds of simulation
            self.signal.update(0.1)
            p = self.signal.current_phase
            if not phases_visited or phases_visited[-1] != p:
                phases_visited.append(p)

        # Visited phases must only be normal cycle phases
        for p in phases_visited:
            self.assertIn(
                p,
                (
                    CyclePhase.NS_GREEN_EW_RED,
                    CyclePhase.NS_YELLOW_EW_RED,
                    CyclePhase.NS_RED_EW_GREEN,
                    CyclePhase.NS_RED_EW_YELLOW,
                ),
            )
        self.assertNotIn(CyclePhase.EMERGENCY_TERMINATING, phases_visited)
        self.assertNotIn(CyclePhase.RECOVERY_ALL_RED, phases_visited)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 8: Civilian Traffic Safety After Recovery
    # ─────────────────────────────────────────────────────────────────────────
    def test_08_civilian_traffic_safety_after_recovery(self):
        """Civilian vehicles obey signals during and after recovery."""
        # Recover to NS_RED_EW_GREEN
        self.signal.current_phase = CyclePhase.NS_RED_EW_GREEN
        self.signal.emergency_terminating = False

        # Spawn civilian vehicle on South approach (facing RED)
        veh_south = CivilianVehicle(
            vehicle_id="CIV-S01",
            approach="SOUTH",
            x=592.0,
            y=550.0,
            speed=50.0,
        )
        self.traffic.vehicles.append(veh_south)

        # Spawn civilian vehicle on East approach (facing GREEN)
        veh_east = CivilianVehicle(
            vehicle_id="CIV-E01",
            approach="EAST",
            x=680.0,
            y=385.0,
            speed=50.0,
        )
        self.traffic.vehicles.append(veh_east)

        # Step simulation
        for _ in range(60):
            self.traffic.update(
                dt=0.05,
                signal_controller=self.signal,
                sim_time=0.0,
            )

        # South vehicle must obey RED and stop before stop line (stop line at y=480.0)
        self.assertGreaterEqual(veh_south.y, 480.0 - 5.0)
        self.assertLessEqual(veh_south.speed, 5.0)

        # East vehicle must proceed on GREEN
        self.assertGreater(veh_east.speed, 20.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 9: Ambulance Natural Forward Continuation After Clearance
    # ─────────────────────────────────────────────────────────────────────────
    def test_09_ambulance_continues_naturally_after_clearance(self):
        """After clearing the intersection, AMB-01 continues moving North without stopping or reversing."""
        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=330.0,  # Fully cleared northern boundary (exit at 350.0)
            speed=70.0,
        )
        self.traffic.ambulance = amb
        amb.cleared = True
        amb.cleared_event_fired = True
        amb.target_speed = 85.0
        lc = self.traffic.lane_coords["SOUTH"]

        initial_y = amb.y
        for _ in range(60):
            amb.update_control(
                dt=0.05,
                signal_state=SignalState.RED,  # Signal is red for South, but amb is already cleared!
                stop_line_coord=lc["stop"],
                intersection_enter=lc["enter"],
                intersection_exit=lc["exit"],
                lead_vehicle=None,
            )
            # Speed must stay positive and forward
            self.assertGreater(amb.speed, 0.0)

        # Traveled north (decreased Y)
        self.assertLess(amb.y, initial_y - 50.0)
        self.assertFalse(amb.braking)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 10: RSU Priority Status Lifecycle
    # ─────────────────────────────────────────────────────────────────────────
    def test_10_rsu_priority_status_lifecycle(self):
        """Verify RSU priority status transitions: INACTIVE -> ACTIVE -> TERMINATING -> INACTIVE."""
        # Initial: INACTIVE
        self.assertEqual(self.signal.rsu_priority_status, "INACTIVE")

        # Emergency preemption active: ACTIVE
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        self.signal.preemption_active = True
        self.assertEqual(self.signal.rsu_priority_status, "ACTIVE")

        # Clearance notification received: TERMINATING
        self.signal.notify_ambulance_cleared("AMB-01")
        self.assertEqual(self.signal.rsu_priority_status, "TERMINATING")
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_TERMINATING)

        # In recovery all-red: still TERMINATING
        self.signal.update(3.1)
        self.assertEqual(self.signal.current_phase, CyclePhase.RECOVERY_ALL_RED)
        self.assertEqual(self.signal.rsu_priority_status, "TERMINATING")

        # Recovered to normal cycle: INACTIVE
        self.signal.update(1.1)
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_RED_EW_GREEN)
        self.assertEqual(self.signal.rsu_priority_status, "INACTIVE")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 11: Headless End-to-End Simulation Lifecycle (Section 26)
    # ─────────────────────────────────────────────────────────────────────────
    def test_11_headless_end_to_end_lifecycle(self):
        """Run a deterministic headless simulation long enough to capture:
        NORMAL CYCLE -> EMERGENCY REQUEST -> PREEMPTION -> SOUTH GREEN ->
        AMBULANCE CROSSING -> FULL CLEARANCE -> EMERGENCY TERMINATION ->
        SOUTH YELLOW -> ALL RED -> NORMAL CYCLE.
        Record timestamps and state transitions. Verify no safety violation.
        """
        sim_time = 0.0
        dt = 1.0 / 60.0
        rsu_pos = (560.0, 415.0)

        # Set phase in NS_RED_EW_GREEN so preemption yellow must be tested
        self.signal.current_phase = CyclePhase.NS_RED_EW_GREEN

        amb = AmbulanceVehicle(
            vehicle_id="AMB-01",
            approach="SOUTH",
            x=592.0,
            y=680.0,
            speed=90.0,
        )
        self.traffic.ambulance = amb
        self.traffic._ambulance_spawned = True

        transition_log = []
        last_phase = self.signal.current_phase
        transition_log.append((sim_time, last_phase.value))

        # Run up to 30 seconds of simulation
        cleared_observed = False
        normal_recovered = False

        for step in range(1800):
            sim_time += dt

            # 1. Update traffic & ambulance
            self.traffic.update(
                dt=dt,
                signal_controller=self.signal,
                sim_time=sim_time,
                v2i_channel=self.v2i_channel,
                rsu_pos=rsu_pos,
            )

            # 2. In-flight message delivery
            delivered = self.v2i_channel.deliver_to_rsu(sim_time)
            for msg in delivered:
                self.signal.receive_emergency_request(msg, current_time=sim_time, auto_preempt=True)

            # 3. Continuous RSU conflict analysis
            self.signal.update_traffic_analysis(
                civilian_vehicles=self.traffic.vehicles,
                ambulance=self.traffic.ambulance,
                current_time=sim_time,
            )

            # 4. Signal controller update
            self.signal.update(dt)

            # Verify mutual exclusion at every tick
            ns_green = (
                self.signal.get_signal("NORTH") == SignalState.GREEN
                or self.signal.get_signal("SOUTH") == SignalState.GREEN
            )
            ew_green = (
                self.signal.get_signal("EAST") == SignalState.GREEN
                or self.signal.get_signal("WEST") == SignalState.GREEN
            )
            self.assertFalse(ns_green and ew_green, f"MUTEX violation at {sim_time:.2f}s in {self.signal.current_phase}")

            # Track transitions
            if self.signal.current_phase != last_phase:
                last_phase = self.signal.current_phase
                transition_log.append((round(sim_time, 2), last_phase.value))

            if "AMB-01 CLEARED INTERSECTION" in self.traffic.ambulance_cleared_events:
                cleared_observed = True

            if cleared_observed and self.signal.current_phase == CyclePhase.NS_RED_EW_GREEN:
                normal_recovered = True
                break

        # Verify full lifecycle was observed
        self.assertTrue(cleared_observed, "AMB-01 should have fully cleared the intersection")
        self.assertTrue(normal_recovered, "Signal should have safely recovered to normal cycle (NS_RED_EW_GREEN)")

        phase_names_visited = [p[1] for p in transition_log]
        self.assertIn("PREEMPTION_YELLOW", phase_names_visited)
        self.assertIn("PREEMPTION_ALL_RED", phase_names_visited)
        self.assertIn("EMERGENCY_SOUTH_GREEN", phase_names_visited)
        self.assertIn("EMERGENCY_TERMINATING", phase_names_visited)
        self.assertIn("RECOVERY_ALL_RED", phase_names_visited)
        self.assertIn("NS_RED_EW_GREEN", phase_names_visited)

        # Check sequence order
        idx_sg = phase_names_visited.index("EMERGENCY_SOUTH_GREEN")
        idx_term = phase_names_visited.index("EMERGENCY_TERMINATING")
        idx_rec_red = phase_names_visited.index("RECOVERY_ALL_RED")
        idx_normal = phase_names_visited.index("NS_RED_EW_GREEN", idx_rec_red)

        self.assertLess(idx_sg, idx_term, "EMERGENCY_SOUTH_GREEN must precede EMERGENCY_TERMINATING")
        self.assertLess(idx_term, idx_rec_red, "EMERGENCY_TERMINATING must precede RECOVERY_ALL_RED")
        self.assertLess(idx_rec_red, idx_normal, "RECOVERY_ALL_RED must precede normal cycle recovery")


if __name__ == "__main__":
    unittest.main()

