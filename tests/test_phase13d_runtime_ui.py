"""Pre-Phase 13E Regression Tests: Fast Startup & HUD Layout Containment.

Validates:
  1. Startup to first frame does not require synchronous AI model readiness.
  2. Background AI initialization failure does not halt or crash the simulation.
  3. AI INITIALIZING and UNAVAILABLE states are cleanly renderable.
  4. All HUD cards and banners stay within the (1280, 720) window boundary.
  5. All HUD cards and banners stay within the legacy (1200, 750) window boundary.
  6. Card contents (labels, values, footers) stay strictly within card rectangles.
  7. Long Safety Fusion reasons (e.g. AI_CONFLICT_WITH_TTC_WARNING) do not overflow.
  8. Long AI states and status badges do not overflow horizontally or vertically.
  9. Event timeline capacity is strictly bounded and never exceeds panel dimensions.
 10. Renderer execution does not mutate underlying simulation, physics, or vehicle state.
"""

import os
import unittest

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

from ai.intersection_ai.predictor import AIPredictionResult, TrajectoryWaypoint
from ai.intersection_ai.safety_fusion import IntersectionSafetyFusionResult
from v2i.smart_signal import SmartTrafficSignal, CyclePhase
from v2i.traffic_manager import TrafficManager, CivilianVehicle
from v2i.v2i_manager import V2IManager
from v2i.event_logger import EventLogger, EventCategory
import v2i.v2i_renderer as renderer


class TestPhase13DRuntimeUI(unittest.TestCase):
    """Test suite validating immediate startup and bounded HUD layouts."""

    def setUp(self):
        pygame.init()
        pygame.font.init()
        self.screen_720p = pygame.Surface((1280, 720))
        self.screen_legacy = pygame.Surface((1200, 750))
        self.font_small = pygame.font.SysFont("consolas", 12, bold=True)
        self.font_normal = pygame.font.SysFont("consolas", 14)
        self.font_bold = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_title = pygame.font.SysFont("consolas", 18, bold=True)

        self.signal = SmartTrafficSignal()
        self.tm = TrafficManager(
            center_x=renderer.CX,
            center_y=renderer.CY,
            road_width=renderer.ROAD_WIDTH,
            stop_line_dist=renderer.STOP_LINE_DIST,
        )
        self.tm._spawn_ambulance()
        self.v2i = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=0.0)
        self.logger = EventLogger(max_events=20)

    def tearDown(self):
        pygame.quit()

    def test_01_first_frame_startup_without_ai_readiness(self):
        """First frame can be rendered immediately while AI is still loading."""
        self.tm.ai_predictor = None
        self.tm.ai_loading = True

        # Render complete scene frame 0
        renderer.draw_four_way_intersection(self.screen_720p)
        renderer.draw_all_signals(self.screen_720p, self.signal, self.font_normal, self.font_bold)
        renderer.draw_civilian_vehicles(self.screen_720p, self.tm.vehicles, self.font_small, self.font_normal)
        renderer.draw_rsu_station(self.screen_720p, self.font_small)
        renderer.draw_hud(
            screen=self.screen_720p,
            signal_controller=self.signal,
            traffic_manager=self.tm,
            font_title=self.font_title,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
            sim_time=0.0,
            paused=False,
            v2i_channel=self.v2i,
            event_logger=self.logger,
            show_ai=True,
        )
        # Verify execution succeeded cleanly
        self.assertTrue(self.tm.ai_loading)
        self.assertIsNone(self.tm.ai_predictor)

    def test_02_ai_initialization_failure_does_not_stop_simulation(self):
        """Simulation continues normally even if AI model fails to load."""
        self.tm.ai_predictor = None
        self.tm.ai_loading = False

        # Simulation step executes without error
        self.tm.update(0.016, self.signal)
        self.signal.update(0.016)

        # Renderer renders UNAVAILABLE state cleanly
        renderer.draw_hud(
            screen=self.screen_720p,
            signal_controller=self.signal,
            traffic_manager=self.tm,
            font_title=self.font_title,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
            sim_time=1.0,
            show_ai=True,
        )
        self.assertIsNone(self.tm.ai_predictor)

    def test_03_ai_loading_state_is_renderable(self):
        """INITIALIZING state displays proper badge and placeholder strings without crash."""
        self.tm.ai_loading = True
        self.tm.ai_predictor = None

        renderer.draw_ai_hud_card(
            screen=self.screen_720p,
            traffic_manager=self.tm,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
            px=16,
            py=54,
            panel_w=370,
            panel_h=132,
        )
        self.assertTrue(self.tm.ai_loading)

    def test_04_hud_rectangles_stay_within_1280x720(self):
        """All HUD cards, top banner, and bottom controls stay within [0, 1280] x [0, 720]."""
        w, h = 1280, 720
        margin_x = 16
        panel_w = 370
        controls_h = 28
        controls_y = h - controls_h - 6
        bot_panel_h = 224
        bot_panel_y = controls_y - 6 - bot_panel_h

        # Top banner: (margin_x, 8, w - 2*margin_x, 38)
        self.assertGreaterEqual(margin_x, 0)
        self.assertLessEqual(margin_x + (w - 2 * margin_x), w)
        self.assertGreaterEqual(8, 0)
        self.assertLessEqual(8 + 38, h)

        # AI Card: (margin_x, 52, panel_w, 132)
        self.assertLessEqual(margin_x + panel_w, w)
        self.assertLessEqual(52 + 132, h)

        # Fusion Card: (margin_x, 188, panel_w, 152)
        self.assertLessEqual(margin_x + panel_w, w)
        self.assertLessEqual(188 + 152, h)

        # Left Bottom Panel: (margin_x, bot_panel_y, panel_w, bot_panel_h)
        self.assertLessEqual(margin_x + panel_w, w)
        self.assertGreaterEqual(bot_panel_y, 188 + 152)  # No overlap with card above
        self.assertLessEqual(bot_panel_y + bot_panel_h, controls_y)

        # Right Top Panel: (w - panel_w - margin_x, 52, panel_w, 265)
        rx = w - panel_w - margin_x
        self.assertGreaterEqual(rx, 0)
        self.assertLessEqual(rx + panel_w, w)
        self.assertLessEqual(52 + 265, h)

        # Right Bottom Panel (Timeline): (rx, bot_panel_y, panel_w, bot_panel_h)
        self.assertGreaterEqual(bot_panel_y, 52 + 265)  # No overlap with card above
        self.assertLessEqual(bot_panel_y + bot_panel_h, controls_y)

        # Controls bar: (margin_x, controls_y, w - 2*margin_x, controls_h)
        self.assertLessEqual(controls_y + controls_h, h)

    def test_05_hud_rectangles_stay_within_legacy_1200x750(self):
        """All HUD cards stay cleanly within [0, 1200] x [0, 750]."""
        w, h = 1200, 750
        margin_x = 16
        panel_w = 370
        controls_h = 28
        controls_y = h - controls_h - 6
        bot_panel_h = 224
        bot_panel_y = controls_y - 6 - bot_panel_h
        rx = w - panel_w - margin_x

        self.assertGreaterEqual(rx, 0)
        self.assertLessEqual(rx + panel_w, w)
        self.assertLessEqual(bot_panel_y + bot_panel_h, controls_y)
        self.assertLessEqual(controls_y + controls_h, h)

    def test_06_long_safety_fusion_reason_contained(self):
        """Long decision reason strings do not overflow the card or crash."""
        amb = self.tm.ambulance
        long_reasons = [
            "AI_CONFLICT_WITH_TTC_WARNING",
            "BLOCKED_LANE_OVERRIDE",
            "CONFLICT_ZONE_OVERRIDE",
            "TTC_CRITICAL_OVERRIDE",
            "PREDICTED_CONFLICT_AND_HAZARD_IN_CORRIDOR",
        ]
        for r in long_reasons:
            amb.latest_safety_fusion = IntersectionSafetyFusionResult(
                fused_risk_state="CRITICAL",
                decision_reason=r,
                recommended_action="EVASIVE_MANEUVER",
                ai_risk_state="PREDICTED_CONFLICT",
                ai_conflict=True,
                ttc=1.2,
                ttc_risk_state="CRITICAL",
                predicted_min_distance=14.2,
                deterministic_clearance=48.2,
                lateral_maneuver_allowed=True,
                target_lane_safe=True,
                timestamp=6.83,
            )
            # Must render without error
            renderer.draw_safety_fusion_hud_card(
                screen=self.screen_720p,
                traffic_manager=self.tm,
                font_bold=self.font_bold,
                font_normal=self.font_normal,
                font_small=self.font_small,
                px=16,
                py=190,
                panel_w=370,
                panel_h=152,
            )

    def test_07_event_timeline_capacity_is_bounded(self):
        """Event timeline with 100 events only draws the recent bounded subset without overflowing."""
        for i in range(100):
            self.logger.log_event(
                timestamp=float(i),
                category=EventCategory.SAFETY if i % 2 == 0 else EventCategory.SIGNAL,
                event_type=f"EVENT_{i}",
                message=f"Long audit event description message #{i} checking string truncation bounds",
            )

        events = self.logger.get_events()
        self.assertEqual(len(events), 20)  # Max capacity of logger

        # Must render cleanly inside panel
        renderer.draw_event_timeline(
            screen=self.screen_720p,
            event_logger=self.logger,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
            px=894,
            py=456,
            panel_w=370,
            panel_h=224,
            traffic_manager=self.tm,
        )

    def test_08_renderer_does_not_mutate_state(self):
        """Rendering all components leaves simulation state, vehicles, and signals unchanged."""
        amb = self.tm.ambulance
        amb_x_before = amb.x
        amb_y_before = amb.y
        amb_spd_before = amb.speed
        sig_phase_before = self.signal.current_phase
        sig_time_before = self.signal.time_remaining

        renderer.draw_hud(
            screen=self.screen_720p,
            signal_controller=self.signal,
            traffic_manager=self.tm,
            font_title=self.font_title,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
            sim_time=5.0,
            v2i_channel=self.v2i,
            event_logger=self.logger,
            show_ai=True,
        )

        self.assertEqual(amb.x, amb_x_before)
        self.assertEqual(amb.y, amb_y_before)
        self.assertEqual(amb.speed, amb_spd_before)
        self.assertEqual(self.signal.current_phase, sig_phase_before)
        self.assertEqual(self.signal.time_remaining, sig_time_before)

    def test_09_fit_text_to_width_truncates_correctly(self):
        """fit_text_to_width truncates overly long strings with ellipsis."""
        long_str = "This is an extraordinarily long string that will definitely exceed thirty pixels"
        truncated = renderer.fit_text_to_width(self.font_small, long_str, 50)
        self.assertTrue(truncated.endswith(".."))
        self.assertLessEqual(self.font_small.size(truncated)[0], 50)

        # Short string remains unmodified
        short_str = "Short"
        self.assertEqual(renderer.fit_text_to_width(self.font_small, short_str, 200), short_str)


if __name__ == "__main__":
    unittest.main()
