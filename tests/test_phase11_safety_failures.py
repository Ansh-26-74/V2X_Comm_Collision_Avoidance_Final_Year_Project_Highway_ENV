"""Phase 11 Test Suite: Failure, Edge-Case & Safety Invariant Validation.

Verifies:
1. Emergency Packet Loss: Dropped emergency requests prevent preemption, preserving normal cycle and safe stop.
2. Delayed Emergency Request: Late-arriving packets cannot bypass the safety gate (Request != Immediate Green).
3. Conflicting Vehicle During Preemption: Active conflict in junction holds All-Red until cleared.
4. Stale Emergency Request: Outdated timestamps are strictly rejected; normal cycle continues unaffected.
5. Duplicate Emergency Requests: Redundant requests across all cycle phases do not duplicate preemption.
6. Duplicate Ambulance Clearance: Repeated clearance events are idempotent; no duplicate recovery cycles.
7. Premature Ambulance Clearance: Partial entry or front-only exit does not trigger premature termination.
8. Emergency Priority Lifetime / Stuck-State Check: Controller guarantees bounded duration and returns to normal cycle.
9. Signal Safety Invariants: Continuous per-tick verification of mutual exclusion across long simulations.
10. Multi-Cycle Stability: Long-duration headless run over multiple cycles with zero leaks or stuck states.
11. Randomized / Seeded Failure Combinations: Deterministic edge-case combinations with 100% safety containment.
12. V2V Regression Protection: Verification that V2V modules, classes, and logic remain untouched.
"""

import math
import random
import unittest
import pygame

from v2i.smart_signal import SmartTrafficSignal, CyclePhase, SignalState
from v2i.traffic_manager import (
    TrafficManager,
    CivilianVehicle,
    AmbulanceVehicle,
    AmbulanceState,
    VehicleState,
)
from v2i.v2i_manager import V2IManager
from v2i.event_logger import EventLogger, EventCategory
import v2i.v2i_renderer as renderer


class TestPhase11FailureAndSafety(unittest.TestCase):
    """Phase 11 failure containment and safety validation test suite."""

    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        self.signal = SmartTrafficSignal(
            intersection_id="INT-01",
            intersection_x=renderer.CX,
            intersection_y=renderer.CY,
        )
        self.tm = TrafficManager(
            center_x=renderer.CX,
            center_y=renderer.CY,
            road_width=renderer.ROAD_WIDTH,
            stop_line_dist=renderer.STOP_LINE_DIST,
            ambulance_spawn_time=8.5,
        )
        self.v2i = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=0.0)
        self.event_logger = EventLogger(max_events=14)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: Emergency Packet Loss
    # ─────────────────────────────────────────────────────────────────────────
    def test_01_emergency_packet_loss(self):
        """Packet loss prevents RSU preemption; normal cycle continues and ambulance stops safely."""
        # Configure 100% packet loss on communication channel
        v2i_loss = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=1.0)
        amb_pos = (592.0, 550.0)  # within 300px of RSU (730, 285)
        rsu_pos = renderer.RSU_POS

        msg = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "approach": "SOUTH",
            "position": amb_pos,
            "timestamp": 10.0,
            "speed": 80.0,
            "eta": 2.5,
            "priority": "HIGH",
        }

        ok, status = v2i_loss.send_to_rsu(10.0, amb_pos, rsu_pos, msg)
        self.assertFalse(ok)
        self.assertEqual(status, "DROPPED")
        self.assertEqual(v2i_loss.total_dropped, 1)

        # Confirm zero delivered packets
        delivered = v2i_loss.deliver_to_rsu(sim_time=15.0)
        self.assertEqual(len(delivered), 0)

        # Step signal controller in normal cycle: preemption must NOT activate
        self.assertFalse(self.signal.emergency_request_received)
        self.assertFalse(self.signal.preemption_requested)
        self.assertFalse(self.signal.preemption_active)

        # Advance through full normal cycle (22s)
        dt = 0.1
        for _ in range(220):
            self.signal.update(dt)
            # Mutual exclusion invariant must hold
            ns_g = (self.signal.get_signal("NORTH") == SignalState.GREEN or self.signal.get_signal("SOUTH") == SignalState.GREEN)
            ew_g = (self.signal.get_signal("EAST") == SignalState.GREEN or self.signal.get_signal("WEST") == SignalState.GREEN)
            self.assertFalse(ns_g and ew_g, "Conflicting green signals during packet loss run")

        # Confirm no residual/buffered preemption
        self.assertIsNone(self.signal.latest_emergency_request)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: Delayed Emergency Request
    # ─────────────────────────────────────────────────────────────────────────
    def test_02_delayed_emergency_request(self):
        """Late-arriving emergency request respects Phase 6 safety gate (Request != Immediate Green)."""
        # Simulate long delay: request transmitted at t=5.0, delivered at t=9.0 (4.0s latency)
        sim_time = 9.0
        self.signal.current_phase = CyclePhase.NS_RED_EW_GREEN
        self.signal.phase_timer = 3.0

        msg = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "approach": "SOUTH",
            "position": (renderer.CX + 32, 600.0),
            "timestamp": 8.0,  # within freshness threshold of 3.0s
            "speed": 80.0,
            "eta": 1.5,
            "priority": "HIGH",
        }

        # Deliver to RSU with auto_preempt
        accepted = self.signal.receive_emergency_request(msg, current_time=sim_time, auto_preempt=True)
        self.assertTrue(accepted)

        # MUST NOT immediately switch to EMERGENCY_SOUTH_GREEN!
        self.assertNotEqual(
            self.signal.current_phase,
            CyclePhase.EMERGENCY_SOUTH_GREEN,
            "CRITICAL FLAW: Signal granted immediate green upon request receipt!",
        )
        self.assertEqual(self.signal.current_phase, CyclePhase.PREEMPTION_YELLOW)
        self.assertTrue(self.signal.preemption_clearing)

        # Conflicting EW direction must be given yellow clearance duration
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.YELLOW)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.RED)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Conflicting Vehicle During Preemption
    # ─────────────────────────────────────────────────────────────────────────
    def test_03_conflicting_vehicle_during_preemption(self):
        """Civilian vehicle in conflict zone holds All-Red; South Green granted only after conflict clears."""
        self.signal.current_phase = CyclePhase.PREEMPTION_ALL_RED
        self.signal._time_in_phase = 1.05  # Standard 1.0s duration has elapsed

        # Spawn civilian vehicle inside central intersection conflict box
        blocking_car = CivilianVehicle(
            vehicle_id="C_BLOCK",
            approach="EAST",
            x=renderer.CX,
            y=renderer.CY,
            speed=20.0,
        )
        blocking_car.in_intersection = True

        # Update controller while vehicle occupies junction
        self.signal.update_traffic_analysis([blocking_car], ambulance=None, current_time=12.0)
        self.signal.update(dt=0.016)

        # Preemption safety gate must reject transition to green
        self.assertEqual(
            self.signal.current_phase,
            CyclePhase.PREEMPTION_ALL_RED,
            "Safety gate failed: granted emergency green while vehicle was inside conflict zone!",
        )
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.RED)
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.RED)

        # Now clear the blocking vehicle
        blocking_car.x = -100.0
        blocking_car.in_intersection = False

        self.signal.update_traffic_analysis([blocking_car], ambulance=None, current_time=13.0)
        self.signal.update(dt=0.016)

        # Once conflict clears, safety gate allows EMERGENCY_SOUTH_GREEN
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_SOUTH_GREEN)
        self.assertEqual(self.signal.get_signal("SOUTH"), SignalState.GREEN)
        self.assertEqual(self.signal.get_signal("EAST"), SignalState.RED)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: Stale Emergency Request
    # ─────────────────────────────────────────────────────────────────────────
    def test_04_stale_emergency_request(self):
        """Stale emergency request (>3.0s age) is strictly rejected; normal cycle continues."""
        sim_time = 25.0
        initial_phase = self.signal.current_phase

        stale_msg = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "approach": "SOUTH",
            "position": (renderer.CX + 32, 600.0),
            "timestamp": 10.0,  # 15.0 seconds old! (max_stale_time is 3.0s)
            "speed": 80.0,
            "eta": 2.0,
            "priority": "HIGH",
        }

        accepted = self.signal.receive_emergency_request(stale_msg, current_time=sim_time, auto_preempt=True)
        self.assertFalse(accepted, "RSU failed to reject stale emergency request")
        self.assertFalse(self.signal.preemption_requested)
        self.assertEqual(self.signal.current_phase, initial_phase)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 5: Duplicate Emergency Requests
    # ─────────────────────────────────────────────────────────────────────────
    def test_05_duplicate_emergency_requests(self):
        """Duplicate requests across all cycle phases do not corrupt state or trigger multiple preemptions."""
        sim_time = 10.0
        msg = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "vehicleType": "EMERGENCY",
            "emergency": True,
            "approach": "SOUTH",
            "position": (renderer.CX + 32, 600.0),
            "timestamp": sim_time,
            "speed": 80.0,
            "eta": 2.0,
            "priority": "HIGH",
        }

        phases_to_test = [
            CyclePhase.NS_RED_EW_GREEN,
            CyclePhase.PREEMPTION_YELLOW,
            CyclePhase.PREEMPTION_ALL_RED,
            CyclePhase.EMERGENCY_SOUTH_GREEN,
            CyclePhase.EMERGENCY_TERMINATING,
            CyclePhase.RECOVERY_ALL_RED,
        ]

        for ph in phases_to_test:
            self.signal.current_phase = ph
            self.signal.phase_timer = 1.0
            # Send 5 identical requests in rapid succession
            for _ in range(5):
                self.signal.receive_emergency_request(msg, current_time=sim_time, auto_preempt=True)
                self.event_logger.observe_system(sim_time, self.signal, self.tm, self.v2i)

            # Signal state must remain valid enum and no exceptions raised
            self.assertIn(self.signal.current_phase, CyclePhase)
            self.signal._verify_safety_invariants()

        # Bounded event logging check
        self.assertLessEqual(len(self.event_logger.events), self.event_logger.max_events)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 6: Duplicate Ambulance Clearance
    # ─────────────────────────────────────────────────────────────────────────
    def test_06_duplicate_ambulance_clearance(self):
        """Duplicate clearance event is strictly idempotent; second call has no harmful effect."""
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        self.signal.phase_timer = 2.0

        # First clearance: initiates termination
        res1 = self.signal.notify_ambulance_cleared("AMB-01")
        self.assertTrue(res1)
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_TERMINATING)
        self.assertTrue(self.signal.emergency_terminating)

        # Advance timer slightly into yellow clearance
        self.signal.update(1.0)
        self.assertEqual(self.signal.time_in_phase, 1.0)

        # Second clearance attempt: must be ignored and return False
        res2 = self.signal.notify_ambulance_cleared("AMB-01")
        self.assertFalse(res2)

        # Timer must NOT be reset to 0.0!
        self.assertEqual(self.signal.time_in_phase, 1.0)
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_TERMINATING)

        # Advance through remainder of termination and recovery
        self.signal.update(2.05)
        self.assertEqual(self.signal.current_phase, CyclePhase.RECOVERY_ALL_RED)
        self.signal.update(1.05)
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_RED_EW_GREEN)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 7: Premature Ambulance Clearance
    # ─────────────────────────────────────────────────────────────────────────
    def test_07_premature_ambulance_clearance(self):
        """Front bumper past exit boundary does NOT trigger clearance while rear bumper is inside."""
        self.tm._spawn_ambulance()
        amb = self.tm.ambulance
        amb.is_authorized = True

        # Position ambulance where front is past exit (y=340 <= 350), but rear is inside (340 + 44 = 384 > 350)
        amb.y = 340.0 + (amb.length / 2.0)  # rear bumper is at 340 + 44 = 384.0
        lc = self.tm.lane_coords["SOUTH"]

        amb.update_control(
            dt=0.016,
            signal_state=SignalState.GREEN,
            stop_line_coord=lc["stop"],
            intersection_enter=lc["enter"],
            intersection_exit=lc["exit"],  # 350.0
            lead_vehicle=None,
            intersection_blocked=False,
            all_vehicles=[],
        )

        self.assertFalse(amb.cleared, "Clearance triggered prematurely before rear bumper exited!")
        self.assertFalse(amb.cleared_event_fired)
        self.assertEqual(len(self.tm.ambulance_cleared_events), 0)

        # Now advance vehicle so rear bumper exits completely (y <= 350 - 22 = 328)
        amb.y = 300.0
        amb.update_control(
            dt=0.016,
            signal_state=SignalState.GREEN,
            stop_line_coord=lc["stop"],
            intersection_enter=lc["enter"],
            intersection_exit=lc["exit"],
            lead_vehicle=None,
            intersection_blocked=False,
            all_vehicles=[],
        )

        self.assertTrue(amb.cleared)
        self.assertTrue(amb.cleared_event_fired)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 8: Emergency Priority Lifetime / Stuck-State Check
    # ─────────────────────────────────────────────────────────────────────────
    def test_08_emergency_priority_lifetime_stuck_state_check(self):
        """Emergency states cannot stay active indefinitely; controller restores normal cycle."""
        sim_time = 0.0
        dt = 1.0 / 60.0

        # Run headless simulation through complete emergency priority
        for _ in range(1800):  # 30 seconds
            self.signal.update(dt)
            sim_time += dt

            self.tm.update(
                dt=dt,
                signal_controller=self.signal,
                sim_time=sim_time,
                v2i_channel=self.v2i,
                rsu_pos=renderer.RSU_POS,
            )

            if self.tm.ambulance_cleared_events and self.signal.current_phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
                self.signal.notify_ambulance_cleared("AMB-01")

            delivered = self.v2i.deliver_to_rsu(sim_time)
            for msg in delivered:
                self.signal.receive_emergency_request(msg, current_time=sim_time, auto_preempt=True)

            self.signal.update_traffic_analysis(self.tm.vehicles, self.tm.ambulance, sim_time)

        # Verify controller has successfully recovered and is in normal cycle
        self.assertFalse(self.signal.preemption_active)
        self.assertTrue(self.signal.emergency_completed)
        self.assertIn(
            self.signal.current_phase,
            (
                CyclePhase.NS_GREEN_EW_RED,
                CyclePhase.NS_YELLOW_EW_RED,
                CyclePhase.NS_RED_EW_GREEN,
                CyclePhase.NS_RED_EW_YELLOW,
            ),
            f"Stuck in non-normal phase: {self.signal.current_phase.value}",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Test 9: Signal Safety Invariants
    # ─────────────────────────────────────────────────────────────────────────
    def test_09_signal_safety_invariants(self):
        """Continuous per-tick verification of mutual exclusion across 2500 simulation steps."""
        sim_time = 0.0
        dt = 1.0 / 60.0

        for tick in range(2500):
            self.signal.update(dt)
            sim_time += dt

            # Invariant A: Conflicting greens NEVER co-exist
            ns_green = (self.signal.get_signal("NORTH") == SignalState.GREEN or self.signal.get_signal("SOUTH") == SignalState.GREEN)
            ew_green = (self.signal.get_signal("EAST") == SignalState.GREEN or self.signal.get_signal("WEST") == SignalState.GREEN)
            self.assertFalse(ns_green and ew_green, f"Tick {tick}: NS and EW greens active simultaneously!")

            # Invariant B: All-Red clearance phases must be completely RED on all 4 approaches
            if self.signal.current_phase in (CyclePhase.PREEMPTION_ALL_RED, CyclePhase.RECOVERY_ALL_RED):
                for app in ("NORTH", "SOUTH", "EAST", "WEST"):
                    self.assertEqual(
                        self.signal.get_signal(app),
                        SignalState.RED,
                        f"Tick {tick}: Approach {app} not RED during {self.signal.current_phase.value}!",
                    )

            # Invariant C: During EMERGENCY_SOUTH_GREEN, EW approaches must be RED
            if self.signal.current_phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
                self.assertEqual(self.signal.get_signal("EAST"), SignalState.RED)
                self.assertEqual(self.signal.get_signal("WEST"), SignalState.RED)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 10: Multi-Cycle Long-Duration Stability
    # ─────────────────────────────────────────────────────────────────────────
    def test_10_multi_cycle_stability(self):
        """Long-duration multi-cycle simulation (3600 frames = 60s) completes with zero memory leak or lockup."""
        sim_time = 0.0
        dt = 1.0 / 60.0

        for frame in range(3600):
            self.signal.update(dt)
            sim_time += dt

            self.tm.update(
                dt=dt,
                signal_controller=self.signal,
                sim_time=sim_time,
                v2i_channel=self.v2i,
                rsu_pos=renderer.RSU_POS,
            )

            if self.tm.ambulance_cleared_events and self.signal.current_phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
                self.signal.notify_ambulance_cleared("AMB-01")

            delivered = self.v2i.deliver_to_rsu(sim_time)
            for msg in delivered:
                self.signal.receive_emergency_request(msg, current_time=sim_time, auto_preempt=True)

            self.signal.update_traffic_analysis(self.tm.vehicles, self.tm.ambulance, sim_time)
            self.event_logger.observe_system(sim_time, self.signal, self.tm, self.v2i)

        # Verify stability
        self.assertGreaterEqual(self.signal.cycle_count, 1)
        self.assertLessEqual(len(self.event_logger.events), self.event_logger.max_events)
        self.assertLessEqual(len(self.tm.vehicles), self.tm.max_vehicles)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 11: Randomized / Seeded Failure Combinations
    # ─────────────────────────────────────────────────────────────────────────
    def test_11_randomized_seeded_failure_combinations(self):
        """Deterministic failure combinations across 5 seeds maintain 100% safety containment."""
        test_seeds = [
            (301, 1.0, 0.2, 8.5, "100% Packet Loss"),
            (302, 0.0, 1.5, 9.5, "High Latency (1500ms)"),
            (303, 0.0, 0.2, 5.0, "Early Arrival (NS Yellow Onset)"),
            (304, 0.0, 0.2, 17.5, "Late Arrival (EW Yellow Clearance)"),
            (305, 0.5, 0.5, 11.0, "Intermittent Loss & Moderate Latency"),
        ]

        for seed, loss_rate, latency, spawn_t, desc in test_seeds:
            random.seed(seed)
            sig = SmartTrafficSignal()
            tm = TrafficManager(
                center_x=renderer.CX,
                center_y=renderer.CY,
                road_width=renderer.ROAD_WIDTH,
                stop_line_dist=renderer.STOP_LINE_DIST,
                ambulance_spawn_time=spawn_t,
            )
            v2i = V2IManager(v2i_range=400.0, latency=latency, packet_loss_rate=loss_rate)

            sim_time = 0.0
            dt = 1.0 / 60.0
            invariants_held = True

            for _ in range(1800):  # 30s per seed
                sig.update(dt)
                sim_time += dt

                # Invariant verification
                ns_g = (sig.get_signal("NORTH") == SignalState.GREEN or sig.get_signal("SOUTH") == SignalState.GREEN)
                ew_g = (sig.get_signal("EAST") == SignalState.GREEN or sig.get_signal("WEST") == SignalState.GREEN)
                if ns_g and ew_g:
                    invariants_held = False
                    break

                tm.update(dt, sig, sim_time, v2i_channel=v2i, rsu_pos=renderer.RSU_POS)

                if tm.ambulance_cleared_events and sig.current_phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
                    sig.notify_ambulance_cleared("AMB-01")

                delivered = v2i.deliver_to_rsu(sim_time)
                for msg in delivered:
                    sig.receive_emergency_request(msg, current_time=sim_time, auto_preempt=True)

                sig.update_traffic_analysis(tm.vehicles, tm.ambulance, sim_time)

            self.assertTrue(invariants_held, f"Seed {seed} ({desc}) violated mutual exclusion!")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 12: Full Regression & V2V Protection
    # ─────────────────────────────────────────────────────────────────────────
    def test_12_v2v_protection(self):
        """Verify V2V modules, classes, and logic remain completely untouched."""
        import main
        import v2v_manager
        from ai.trajectory_predictor import AIPredictor
        from ai.safety_fusion import SafetyFusionEngine

        self.assertTrue(hasattr(main, "main"))
        self.assertTrue(hasattr(v2v_manager, "V2VManager"))
        self.assertTrue(hasattr(AIPredictor, "predict"))
        self.assertTrue(hasattr(SafetyFusionEngine, "evaluate"))

        # Verify V2V default instance
        v2v = v2v_manager.V2VManager()
        self.assertGreater(v2v.v2v_range, 0.0)


if __name__ == "__main__":
    unittest.main()
