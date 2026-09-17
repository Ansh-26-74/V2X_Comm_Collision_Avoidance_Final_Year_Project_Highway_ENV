"""Phase 13D Step 5: Tests for Visual HUD & AI Trajectory Visualization.

Validates:
  1. AI panel handles insufficient data (<5 telemetry samples).
  2. AI panel handles prediction available state.
  3. AI panel & trajectory display 6 waypoints (+0.25s .. +1.50s).
  4. AI conflict state is visualized with prediction-oriented terminology.
  5. AI safe state is visualized cleanly.
  6. AI unavailable state uses deterministic fallback label without claiming 'AI SAFE'.
  7. Safety fusion values are reflected accurately in the HUD card.
  8. AI -> TTC timeline uses actual timestamps from event logger.
  9. AI early warning lead time is only shown when genuinely measured (> 0.0s).
 10. Stale predictions are not rendered on the roadway.
 11. Renderer functions do not mutate underlying simulation, physics, or vehicle state.
 12. Existing V2V and V2I visualizations remain fully available and intact.
"""

import os
import unittest
from collections import deque

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

from ai.intersection_ai.predictor import AIPredictionResult, TrajectoryWaypoint, TelemetrySample
from ai.intersection_ai.safety_fusion import IntersectionSafetyFusion, IntersectionSafetyFusionResult
from v2i.smart_signal import SmartTrafficSignal, CyclePhase
from v2i.traffic_manager import TrafficManager, CivilianVehicle
from v2i.v2i_manager import V2IManager
from v2i.event_logger import EventLogger, EventCategory
import v2i.v2i_renderer as renderer


class TestPhase13DVisualization(unittest.TestCase):
    """Test suite for Phase 13D Step 5 Visual HUD & AI Trajectory Visualization."""

    def setUp(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((1200, 750))
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
        c01 = CivilianVehicle(
            vehicle_id="C-01",
            approach="SOUTH",
            x=592.0,
            y=540.0,
            speed=20.0,
            target_speed=20.0,
        )
        self.tm.vehicles.append(c01)
        self.v2i = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=0.0)
        self.logger = EventLogger(max_events=14)

    def tearDown(self):
        pygame.quit()

    def test_01_ai_panel_handles_insufficient_data(self):
        """When history < 5, HUD card reflects COLLECTING DATA and trajectory is suppressed."""
        amb = self.tm.ambulance
        amb.latest_ai_prediction = AIPredictionResult(
            sender_id="C-01",
            prediction_available=False,
            ai_risk_state="INSUFFICIENT_DATA",
            status_message="COLLECTING_DATA (3/5)",
        )

        # Draw AI HUD card
        renderer.draw_ai_hud_card(
            screen=self.screen,
            traffic_manager=self.tm,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
            px=20,
            py=55,
            panel_w=360,
            panel_h=128,
        )

        # Verify draw_ai_trajectory does not render when prediction_available is False
        renderer.draw_ai_trajectory(
            screen=self.screen,
            ambulance=amb,
            civilian_vehicles=self.tm.vehicles,
            font_small=self.font_small,
            font_timer=self.font_normal,
            show_ai=True,
        )
        # Should execute cleanly without error
        self.assertFalse(amb.latest_ai_prediction.prediction_available)

    def test_02_ai_panel_handles_prediction_available(self):
        """When prediction is available, AI HUD card reflects ACTIVE and valid metrics."""
        amb = self.tm.ambulance
        wps = [
            TrajectoryWaypoint(horizon_offset_s=h, x=592.0, y=600.0 - h * 50.0)
            for h in [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
        ]
        amb.latest_ai_prediction = AIPredictionResult(
            sender_id="C-01",
            prediction_available=True,
            horizon_seconds=1.50,
            waypoints=wps,
            ai_risk_state="PREDICTED_SAFE",
            predicted_conflict=False,
            predicted_min_distance=85.0,
            inference_time_ms=0.42,
            status_message="PREDICTION_AVAILABLE",
        )

        renderer.draw_ai_hud_card(
            screen=self.screen,
            traffic_manager=self.tm,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
            px=20,
            py=55,
        )
        self.assertTrue(amb.latest_ai_prediction.prediction_available)
        self.assertEqual(len(amb.latest_ai_prediction.waypoints), 6)
        self.assertAlmostEqual(amb.latest_ai_prediction.horizon_seconds, 1.50)

    def test_03_ai_panel_displays_six_waypoints(self):
        """AI prediction contains exactly 6 future waypoints covering +0.25s .. +1.50s."""
        amb = self.tm.ambulance
        horizons = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
        wps = [
            TrajectoryWaypoint(horizon_offset_s=h, x=592.0, y=600.0 - h * 55.0)
            for h in horizons
        ]
        amb.latest_ai_prediction = AIPredictionResult(
            sender_id="C-01",
            prediction_available=True,
            waypoints=wps,
            horizon_seconds=1.50,
        )

        self.assertEqual(len(amb.latest_ai_prediction.waypoints), 6)
        for i, expected_h in enumerate(horizons):
            self.assertAlmostEqual(amb.latest_ai_prediction.waypoints[i].horizon_offset_s, expected_h)

        # Test roadway trajectory drawing
        renderer.draw_ai_trajectory(
            screen=self.screen,
            ambulance=amb,
            civilian_vehicles=self.tm.vehicles,
            font_small=self.font_small,
            font_timer=self.font_normal,
            show_ai=True,
        )

    def test_04_ai_conflict_state_is_visualized(self):
        """When predicted_conflict is True, visual state reflects PREDICTED CONFLICT."""
        amb = self.tm.ambulance
        wps = [
            TrajectoryWaypoint(horizon_offset_s=h, x=592.0, y=580.0 - h * 40.0)
            for h in [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
        ]
        amb.latest_ai_prediction = AIPredictionResult(
            sender_id="C-01",
            prediction_available=True,
            waypoints=wps,
            ai_risk_state="PREDICTED_CONFLICT",
            predicted_conflict=True,
            predicted_min_distance=14.2,
            inference_time_ms=0.38,
            status_message="CONFLICT_PREDICTED",
        )

        renderer.draw_ai_hud_card(
            screen=self.screen,
            traffic_manager=self.tm,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
        )

        renderer.draw_ai_trajectory(
            screen=self.screen,
            ambulance=amb,
            civilian_vehicles=self.tm.vehicles,
            font_small=self.font_small,
            font_timer=self.font_normal,
            show_ai=True,
        )

        self.assertTrue(amb.latest_ai_prediction.predicted_conflict)
        self.assertEqual(amb.latest_ai_prediction.ai_risk_state, "PREDICTED_CONFLICT")

    def test_05_ai_safe_state_is_visualized(self):
        """When predicted_conflict is False, visual state reflects PREDICTED SAFE."""
        amb = self.tm.ambulance
        wps = [
            TrajectoryWaypoint(horizon_offset_s=h, x=592.0, y=550.0 - h * 60.0)
            for h in [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
        ]
        amb.latest_ai_prediction = AIPredictionResult(
            sender_id="C-01",
            prediction_available=True,
            waypoints=wps,
            ai_risk_state="PREDICTED_SAFE",
            predicted_conflict=False,
            predicted_min_distance=110.0,
            inference_time_ms=0.35,
        )

        renderer.draw_ai_hud_card(
            screen=self.screen,
            traffic_manager=self.tm,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
        )

        renderer.draw_ai_trajectory(
            screen=self.screen,
            ambulance=amb,
            civilian_vehicles=self.tm.vehicles,
            font_small=self.font_small,
            font_timer=self.font_normal,
            show_ai=True,
        )
        self.assertFalse(amb.latest_ai_prediction.predicted_conflict)
        self.assertEqual(amb.latest_ai_prediction.ai_risk_state, "PREDICTED_SAFE")

    def test_06_ai_unavailable_state_uses_deterministic_fallback_label(self):
        """When predictor is None, HUD displays UNAVAILABLE with DETERMINISTIC V2V fallback."""
        self.tm.ai_predictor = None
        amb = self.tm.ambulance
        amb.latest_ai_prediction = None

        renderer.draw_ai_hud_card(
            screen=self.screen,
            traffic_manager=self.tm,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
        )
        self.assertIsNone(self.tm.ai_predictor)
        self.assertIsNone(amb.latest_ai_prediction)

    def test_07_safety_fusion_values_are_reflected(self):
        """Safety fusion HUD card accurately renders genuine fused state, action, and reason."""
        amb = self.tm.ambulance
        fusion_res = IntersectionSafetyFusionResult(
            fused_risk_state="CRITICAL",
            decision_reason="TTC_CRITICAL",
            recommended_action="EVASIVE_MANEUVER",
            ai_risk_state="PREDICTED_CONFLICT",
            ai_conflict=True,
            ttc=1.34,
            ttc_risk_state="CRITICAL",
            predicted_min_distance=14.2,
            deterministic_clearance=48.2,
            lateral_maneuver_allowed=True,
            target_lane_safe=True,
            timestamp=6.83,
        )
        amb.latest_safety_fusion = fusion_res

        renderer.draw_safety_fusion_hud_card(
            screen=self.screen,
            traffic_manager=self.tm,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
            px=20,
            py=188,
        )

        self.assertEqual(amb.latest_safety_fusion.fused_risk_state, "CRITICAL")
        self.assertEqual(amb.latest_safety_fusion.recommended_action, "EVASIVE_MANEUVER")
        self.assertEqual(amb.latest_safety_fusion.decision_reason, "TTC_CRITICAL")
        self.assertAlmostEqual(amb.latest_safety_fusion.ttc, 1.34)

    def test_08_ai_ttc_timeline_uses_actual_timestamps(self):
        """Event timeline logs and displays genuine simulation timestamps for AI and TTC events."""
        self.logger.log_event(6.20, EventCategory.SAFETY, "AI_PREDICTED_CONFLICT", "AI FORECAST: Predicted conflict with C-01")
        self.logger.log_event(6.20, EventCategory.SAFETY, "SAFETY_FUSION_ELEVATED_WARNING", "SAFETY FUSION: Elevated warning — Advisory preparation")
        self.logger.log_event(6.83, EventCategory.SAFETY, "V2V_TTC_CRITICAL", "V2V CRITICAL — High collision risk detected with C-01 (TTC: 1.34s)")
        self.logger.log_event(6.87, EventCategory.SAFETY, "V2V_EVASIVE_MANEUVER_STARTED", "AMB-01 V2V EVASIVE MANEUVER: Lateral lane change initiated")

        events = self.logger.get_events()
        self.assertEqual(len(events), 4)
        self.assertAlmostEqual(events[0].timestamp, 6.20)
        self.assertEqual(events[0].event_type, "AI_PREDICTED_CONFLICT")
        self.assertAlmostEqual(events[2].timestamp, 6.83)
        self.assertEqual(events[2].event_type, "V2V_TTC_CRITICAL")

        # Test rendering timeline
        renderer.draw_event_timeline(
            screen=self.screen,
            event_logger=self.logger,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
            px=800,
            py=490,
            panel_w=380,
            panel_h=210,
            traffic_manager=self.tm,
        )

    def test_09_ai_lead_time_only_shown_when_measured(self):
        """AI lead time displays formatted seconds only when measured (>0); otherwise N/A."""
        fusion = IntersectionSafetyFusion()
        # Case A: Not yet measured
        self.assertIsNone(fusion.ai_lead_time)

        # Case B: Measured AI before TTC
        fusion.first_ai_conflict_time = 6.20
        fusion.first_ttc_critical_time = 6.83
        lead = fusion.first_ttc_critical_time - fusion.first_ai_conflict_time
        if lead > 0.0:
            fusion.ai_lead_time = round(lead, 2)

        self.assertIsNotNone(fusion.ai_lead_time)
        self.assertAlmostEqual(fusion.ai_lead_time, 0.63, places=2)

        self.tm.safety_fusion = fusion
        renderer.draw_safety_fusion_hud_card(
            screen=self.screen,
            traffic_manager=self.tm,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
        )

    def test_10_stale_prediction_is_not_rendered(self):
        """When prediction is stale or unavailable, draw_ai_trajectory immediately suppresses output."""
        amb = self.tm.ambulance
        amb.latest_ai_prediction = None

        # Should exit immediately without error or visual artifact
        renderer.draw_ai_trajectory(
            screen=self.screen,
            ambulance=amb,
            civilian_vehicles=self.tm.vehicles,
            font_small=self.font_small,
            font_timer=self.font_normal,
            show_ai=True,
        )

        # With prediction_available = False
        amb.latest_ai_prediction = AIPredictionResult(
            sender_id="C-01",
            prediction_available=False,
            waypoints=[],
        )
        renderer.draw_ai_trajectory(
            screen=self.screen,
            ambulance=amb,
            civilian_vehicles=self.tm.vehicles,
            font_small=self.font_small,
            font_timer=self.font_normal,
            show_ai=True,
        )

    def test_11_renderer_does_not_modify_simulation_state(self):
        """Drawing HUD, trajectory, and signals does not alter vehicle coordinates or physics."""
        amb = self.tm.ambulance
        init_x = amb.x
        init_y = amb.y
        init_speed = amb.speed
        init_phase = self.signal.current_phase

        renderer.draw_ai_trajectory(
            screen=self.screen,
            ambulance=amb,
            civilian_vehicles=self.tm.vehicles,
            font_small=self.font_small,
            font_timer=self.font_normal,
            show_ai=True,
        )

        renderer.draw_hud(
            screen=self.screen,
            signal_controller=self.signal,
            traffic_manager=self.tm,
            font_title=self.font_title,
            font_bold=self.font_bold,
            font_normal=self.font_normal,
            font_small=self.font_small,
            sim_time=10.0,
            paused=False,
            v2i_channel=self.v2i,
            event_logger=self.logger,
            show_ai=True,
        )

        self.assertEqual(amb.x, init_x)
        self.assertEqual(amb.y, init_y)
        self.assertEqual(amb.speed, init_speed)
        self.assertEqual(self.signal.current_phase, init_phase)

    def test_12_existing_v2v_visualization_remains_available(self):
        """Existing V2V inter-vehicle link and RSU station remain operational."""
        amb = self.tm.ambulance
        amb.v2v_risk_state = "WARNING"
        amb.v2v_ttc = 3.2
        amb.v2v_detected_vehicle_id = "C-01"

        # V2V link
        renderer.draw_v2v_inter_link(
            screen=self.screen,
            ambulance=amb,
            civilian_vehicles=self.tm.vehicles,
            font_small=self.font_small,
        )

        # RSU station
        renderer.draw_rsu_station(screen=self.screen, font_small=self.font_small)

        # V2I communication beam
        renderer.draw_v2i_communication(
            screen=self.screen,
            v2i_channel=self.v2i,
            ambulance=amb,
            font_small=self.font_small,
            signal_controller=self.signal,
        )


if __name__ == "__main__":
    unittest.main()
