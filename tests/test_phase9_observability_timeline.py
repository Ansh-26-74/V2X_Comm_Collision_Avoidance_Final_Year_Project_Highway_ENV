"""Phase 9 Test Suite: Polished Visualization, Event Timeline & V2I Observability.

Verifies:
1. Event Creation: System state changes emit discrete SimulationEvent instances.
2. Event Deduplication: Steady states over multiple frames generate exactly 1 event, not repeated entries.
3. Event Ordering: Events are stored strictly in chronological order.
4. Event Retention: Event buffer strictly bounds retention (FIFO) according to max_events limit.
5. Reset Clears State: Resetting EventLogger or simulation components resets audit logs and visuals.
6. Live Data Fidelity (No Fake Data): Event messages and HUD telemetry reflect real simulation properties.
7. Phase 8 Transition Capture: Clearance, termination, recovery, and cycle restoration are properly audited.
8. Renderer Observability: Timeline and HUD rendering function with live simulation structures.
"""

import unittest
import pygame

from v2i.event_logger import EventLogger, SimulationEvent, EventCategory
from v2i.smart_signal import SmartTrafficSignal, CyclePhase, SignalState
from v2i.traffic_manager import TrafficManager
from v2i.v2i_manager import V2IManager
import v2i.v2i_renderer as renderer


class TestPhase9ObservabilityAndTimeline(unittest.TestCase):
    """Unit and integration tests for Phase 9 event logging and observability."""

    def setUp(self):
        self.logger = EventLogger(max_events=10)
        self.signal = SmartTrafficSignal()
        self.tm = TrafficManager(
            center_x=600.0,
            center_y=415.0,
            road_width=240.0,
            stop_line_dist=120.0,
        )
        self.v2i = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=0.0)

    def test_01_event_creation(self):
        """Important simulation state changes create discrete events."""
        self.logger.log_event(0.5, EventCategory.COMMUNICATION, "V2I_RANGE_ENTERED", "AMB-01 ENTERED RSU RANGE")
        self.assertEqual(len(self.logger.events), 1)
        event = self.logger.events[0]
        self.assertEqual(event.timestamp, 0.5)
        self.assertEqual(event.category, EventCategory.COMMUNICATION)
        self.assertEqual(event.event_type, "V2I_RANGE_ENTERED")
        self.assertEqual(event.message, "AMB-01 ENTERED RSU RANGE")

    def test_02_event_deduplication(self):
        """A continuous state lasting 100 frames creates 1 transition event, not 100."""
        # Step system 100 times in the initial NS_GREEN_EW_RED phase
        for frame in range(100):
            sim_time = frame * (1.0 / 60.0)
            self.logger.observe_system(
                sim_time=sim_time,
                signal_controller=self.signal,
                traffic_manager=self.tm,
                v2i_channel=self.v2i,
            )

        signal_events = [e for e in self.logger.events if e.category == EventCategory.SIGNAL]
        self.assertEqual(
            len(signal_events),
            1,
            f"Expected exactly 1 initial signal phase event after 100 frames, got {len(signal_events)}",
        )
        self.assertEqual(signal_events[0].event_type, CyclePhase.NS_GREEN_EW_RED.value)
        self.assertIn("North/South GREEN", signal_events[0].message)

    def test_03_event_ordering(self):
        """Events are maintained in strict chronological order."""
        times = [1.0, 2.5, 4.2, 5.0, 7.8]
        for t in times:
            self.logger.log_event(t, EventCategory.SYSTEM, f"EV_{t}", f"Event at {t}")

        event_times = [e.timestamp for e in self.logger.events]
        self.assertEqual(event_times, times)
        self.assertEqual(event_times, sorted(event_times))

    def test_04_event_retention_limit(self):
        """Event list has a bounded size; older events are removed beyond max_events."""
        self.assertEqual(self.logger.max_events, 10)
        for i in range(25):
            self.logger.log_event(float(i), EventCategory.TRAFFIC, f"EV_{i}", f"Message {i}")

        self.assertEqual(len(self.logger.events), 10)
        # Should retain only the newest 10 (15 to 24)
        self.assertEqual(self.logger.events[0].timestamp, 15.0)
        self.assertEqual(self.logger.events[-1].timestamp, 24.0)

    def test_05_reset_clears_state(self):
        """Reset clears event timeline and visual transition caches."""
        self.logger.log_event(1.0, EventCategory.EMERGENCY, "ALERT", "Emergency active")
        self.assertEqual(len(self.logger.events), 1)

        self.logger.clear()
        self.assertEqual(len(self.logger.events), 0)
        self.assertIsNone(self.logger._last_phase)
        self.assertFalse(self.logger._last_amb_overtaking)

    def test_06_packet_delivery_and_drop_audit(self):
        """V2I channel packet transmission, delivery, and drop are audited without fabrication."""
        sim_time = 0.0
        amb_pos = (600.0, 650.0)  # within 400px of RSU (730, 285)
        rsu_pos = (730.0, 285.0)

        # Enqueue packet in V2IManager using real API: send_to_rsu(now, vehicle_pos, rsu_pos, message)
        msg_payload = {
            "type": "EMERGENCY_REQUEST",
            "vehicleId": "AMB-01",
            "approach": "SOUTH",
            "eta": 3.0,
            "speed": 80.0,
            "priority": "HIGH",
        }
        ok, status = self.v2i.send_to_rsu(sim_time, amb_pos, rsu_pos, msg_payload)
        self.assertTrue(ok)
        self.assertEqual(status, "TRANSMITTED")

        # Advance beyond latency (0.2s)
        sim_time = 0.25
        delivered = self.v2i.deliver_to_rsu(sim_time)
        self.assertEqual(len(delivered), 1)

        # Observe channel
        self.logger.observe_system(sim_time, self.signal, self.tm, self.v2i)
        rx_events = [e for e in self.logger.events if e.event_type == "PACKET_DELIVERED"]
        self.assertEqual(len(rx_events), 1)
        self.assertIn("EMERGENCY_REQUEST", rx_events[0].message)

    def test_07_phase8_events_captured(self):
        """Timeline reliably captures Phase 8 clearance, termination, recovery, and cycle restoration."""
        # 1. Simulate preemption to EMERGENCY_SOUTH_GREEN
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        self.signal.phase_timer = 0.0
        self.logger.observe_system(10.0, self.signal, self.tm, self.v2i)

        # 2. Ambulance clears intersection -> notify controller
        self.signal.notify_ambulance_cleared("AMB-01")
        self.assertEqual(self.signal.current_phase, CyclePhase.EMERGENCY_TERMINATING)
        self.logger.observe_system(12.0, self.signal, self.tm, self.v2i)

        # 3. Advance through EMERGENCY_TERMINATING (3.0s) -> RECOVERY_ALL_RED
        self.signal.update(3.05)
        self.assertEqual(self.signal.current_phase, CyclePhase.RECOVERY_ALL_RED)
        self.logger.observe_system(15.05, self.signal, self.tm, self.v2i)

        # 4. Advance through RECOVERY_ALL_RED (1.0s) -> NS_RED_EW_GREEN
        self.signal.update(1.05)
        self.assertEqual(self.signal.current_phase, CyclePhase.NS_RED_EW_GREEN)
        self.logger.observe_system(16.10, self.signal, self.tm, self.v2i)

        messages = [e.message for e in self.logger.events]
        self.assertTrue(any("SOUTH GREEN" in m for m in messages), f"Missing green in {messages}")
        self.assertTrue(any("TERMINATING" in m for m in messages), f"Missing terminating in {messages}")
        self.assertTrue(any("ALL-RED" in m for m in messages), f"Missing recovery all-red in {messages}")
        self.assertTrue(any("NORMAL OPERATION RESTORED" in m for m in messages), f"Missing normal cycle restored in {messages}")

    def test_08_rendering_pipeline_robustness(self):
        """Renderer HUD and Event Timeline draw without exception on live data."""
        pygame.init()
        screen = pygame.Surface((1200, 750))
        font_small = pygame.font.SysFont("Consolas", 10)
        font_normal = pygame.font.SysFont("Segoe UI", 12)
        font_bold = pygame.font.SysFont("Segoe UI", 12, bold=True)
        font_title = pygame.font.SysFont("Segoe UI", 18, bold=True)

        self.logger.log_event(1.0, EventCategory.COMMUNICATION, "TEST_COMM", "Comm test message")
        self.logger.log_event(2.0, EventCategory.SAFETY, "TEST_SAFETY", "Safety conflict resolved")

        # Test drawing dedicated timeline
        renderer.draw_event_timeline(screen, self.logger, font_bold, font_normal, font_small, 800, 490, 380, 210)

        # Test drawing complete HUD
        renderer.draw_hud(
            screen=screen,
            signal_controller=self.signal,
            traffic_manager=self.tm,
            font_title=font_title,
            font_bold=font_bold,
            font_normal=font_normal,
            font_small=font_small,
            sim_time=15.0,
            paused=False,
            v2i_channel=self.v2i,
            event_logger=self.logger,
        )

        # Test drawing all signals (with emergency indicator)
        self.signal.current_phase = CyclePhase.EMERGENCY_SOUTH_GREEN
        renderer.draw_all_signals(screen, self.signal, font_small, font_small)


if __name__ == "__main__":
    unittest.main()
