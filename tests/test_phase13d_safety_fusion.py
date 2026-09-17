"""Phase 13D Step 4: Deterministic Safety Fusion Engine Tests.

Validates:
1. AI predicted conflict + TTC critical -> Fused CRITICAL + EVASIVE_MANEUVER.
2. AI predicted conflict + TTC warning -> Fused ELEVATED_WARNING + PREPARE_EVASION.
3. AI predicted conflict + TTC safe -> Fused ADVISORY_MONITORING + CONTINUE (AI does NOT force maneuver).
4. AI unavailable -> complete deterministic TTC fallback.
5. AI inference error -> complete deterministic TTC fallback.
6. TTC critical remains authoritative even if AI predicts safe.
7. Conflict zone override: lateral evasion barred inside conflict zone.
8. Blocked lane override: target lane occupied -> EMERGENCY_BRAKE action.
9. Fused result is bitwise deterministic across identical inputs.
10. AI prediction alone cannot cause unsafe or unauthorized lateral movement.
11. Measurement instrumentation tracks AI vs TTC lead time accurately.
12. Existing V2V evasive maneuver and clearance behavior remains 100% intact.
"""

import math
import random
import unittest

from ai.intersection_ai.predictor import (
    AIPredictionResult,
    TrajectoryWaypoint,
    IntersectionTrajectoryPredictor,
)
from ai.intersection_ai.safety_fusion import (
    IntersectionSafetyFusion,
    IntersectionSafetyFusionResult,
)
from v2i.smart_signal import SmartTrafficSignal
from v2i.traffic_manager import (
    TrafficManager,
    AmbulanceVehicle,
    CivilianVehicle,
    SOUTH_PRIMARY_LANE_X,
    SOUTH_OVERTAKE_LANE_X,
)
from v2i.v2i_manager import V2IManager
from v2i.v2v_inter_manager import IntersectionV2VManager


class TestPhase13DSafetyFusion(unittest.TestCase):
    """Test suite for Phase 13D Step 4: Deterministic Safety Fusion Engine."""

    def setUp(self):
        """Instantiate safety fusion engine."""
        self.fusion = IntersectionSafetyFusion(
            critical_ttc_threshold=2.0,
            warning_ttc_threshold=4.0,
            critical_gap_threshold=25.0,
        )

    def test_01_ai_conflict_ttc_critical(self):
        """Test 1: AI conflict + TTC critical -> Fused CRITICAL + EVASIVE_MANEUVER."""
        ai_res = AIPredictionResult(
            sender_id="C-01",
            prediction_available=True,
            ai_risk_state="PREDICTED_CONFLICT",
            predicted_conflict=True,
            predicted_min_distance=15.0,
        )

        res = self.fusion.evaluate(
            ai_prediction=ai_res,
            ttc=1.2,
            ttc_risk="CRITICAL",
            deterministic_clearance=45.0,
            lateral_maneuver_allowed=True,
            target_lane_safe=True,
            sim_time=5.0,
        )

        self.assertEqual(res.fused_risk_state, "CRITICAL")
        self.assertEqual(res.decision_reason, "TTC_CRITICAL")
        self.assertEqual(res.recommended_action, "EVASIVE_MANEUVER")

    def test_02_ai_conflict_ttc_warning(self):
        """Test 2: AI conflict + TTC warning -> Fused ELEVATED_WARNING + PREPARE_EVASION."""
        ai_res = AIPredictionResult(
            sender_id="C-01",
            prediction_available=True,
            ai_risk_state="PREDICTED_CONFLICT",
            predicted_conflict=True,
            predicted_min_distance=28.0,
        )

        res = self.fusion.evaluate(
            ai_prediction=ai_res,
            ttc=2.8,
            ttc_risk="WARNING",
            deterministic_clearance=90.0,
            lateral_maneuver_allowed=True,
            target_lane_safe=True,
            sim_time=4.5,
        )

        self.assertEqual(res.fused_risk_state, "ELEVATED_WARNING")
        self.assertEqual(res.decision_reason, "AI_CONFLICT_WITH_TTC_WARNING")
        self.assertEqual(res.recommended_action, "PREPARE_EVASION")

    def test_03_ai_conflict_ttc_safe(self):
        """Test 3: AI conflict + TTC safe -> Fused ADVISORY_MONITORING + CONTINUE (AI does NOT force maneuver)."""
        ai_res = AIPredictionResult(
            sender_id="C-01",
            prediction_available=True,
            ai_risk_state="PREDICTED_CONFLICT",
            predicted_conflict=True,
            predicted_min_distance=35.0,
        )

        # TTC is SAFE (> 4.0s) with 150px gap
        res = self.fusion.evaluate(
            ai_prediction=ai_res,
            ttc=6.5,
            ttc_risk="SAFE",
            deterministic_clearance=150.0,
            lateral_maneuver_allowed=True,
            target_lane_safe=True,
            sim_time=3.5,
        )

        self.assertEqual(res.fused_risk_state, "ADVISORY_MONITORING")
        self.assertEqual(res.decision_reason, "AI_CONFLICT_BUT_TTC_SAFE")
        self.assertEqual(res.recommended_action, "CONTINUE")

    def test_04_ai_unavailable_fallback(self):
        """Test 4: AI unavailable (None) -> 100% deterministic fallback."""
        res = self.fusion.evaluate(
            ai_prediction=None,
            ttc=3.2,
            ttc_risk="WARNING",
            deterministic_clearance=85.0,
            lateral_maneuver_allowed=True,
            target_lane_safe=True,
            sim_time=2.0,
        )

        self.assertEqual(res.decision_reason, "AI_UNAVAILABLE_DETERMINISTIC_FALLBACK")
        self.assertEqual(res.fused_risk_state, "WARNING")
        self.assertEqual(res.recommended_action, "CONTINUE")
        self.assertFalse(res.ai_conflict)

    def test_05_ai_inference_error_fallback(self):
        """Test 5: AI reporting error/unavailable -> safe deterministic fallback."""
        ai_res = AIPredictionResult(
            sender_id="C-01",
            prediction_available=False,
            ai_risk_state="INSUFFICIENT_DATA",
            status_message="INFERENCE_ERROR: CUDA out of memory",
        )

        res = self.fusion.evaluate(
            ai_prediction=ai_res,
            ttc=5.5,
            ttc_risk="SAFE",
            deterministic_clearance=160.0,
            lateral_maneuver_allowed=True,
            target_lane_safe=True,
            sim_time=2.2,
        )

        self.assertEqual(res.decision_reason, "AI_UNAVAILABLE_DETERMINISTIC_FALLBACK")
        self.assertEqual(res.fused_risk_state, "SAFE")
        self.assertEqual(res.recommended_action, "CONTINUE")

    def test_06_ttc_critical_remains_authoritative(self):
        """Test 6: Even if AI predicts SAFE, deterministic TTC CRITICAL forces authoritative action."""
        ai_res = AIPredictionResult(
            sender_id="C-01",
            prediction_available=True,
            ai_risk_state="PREDICTED_SAFE",
            predicted_conflict=False,
            predicted_min_distance=80.0,
        )

        # Physical gap is 20px (imminent crash) and TTC is 0.4s
        res = self.fusion.evaluate(
            ai_prediction=ai_res,
            ttc=0.4,
            ttc_risk="CRITICAL",
            deterministic_clearance=20.0,
            lateral_maneuver_allowed=True,
            target_lane_safe=True,
            sim_time=6.0,
        )

        # Deterministic safety hierarchy overrides AI false-negative
        self.assertEqual(res.fused_risk_state, "CRITICAL")
        self.assertEqual(res.decision_reason, "TTC_CRITICAL")
        self.assertEqual(res.recommended_action, "EVASIVE_MANEUVER")

    def test_07_conflict_zone_safety_override(self):
        """Test 7: AI cannot override conflict zone restrictions (lateral maneuver barred)."""
        ai_res = AIPredictionResult(
            sender_id="C-01",
            prediction_available=True,
            ai_risk_state="PREDICTED_CONFLICT",
            predicted_conflict=True,
            predicted_min_distance=10.0,
        )

        # Inside conflict zone: lateral_maneuver_allowed=False, in_intersection=True
        res = self.fusion.evaluate(
            ai_prediction=ai_res,
            ttc=1.1,
            ttc_risk="CRITICAL",
            deterministic_clearance=30.0,
            lateral_maneuver_allowed=False,
            target_lane_safe=True,
            sim_time=9.0,
            in_intersection=True,
        )

        self.assertEqual(res.decision_reason, "CONFLICT_ZONE_OVERRIDE")
        self.assertEqual(res.recommended_action, "CONTINUE")

    def test_08_blocked_lane_safety_override(self):
        """Test 8: Blocked adjacent lane forces EMERGENCY_BRAKE even if AI recommends evasion."""
        ai_res = AIPredictionResult(
            sender_id="C-01",
            prediction_available=True,
            ai_risk_state="PREDICTED_CONFLICT",
            predicted_conflict=True,
            predicted_min_distance=15.0,
        )

        # Target lane is NOT safe (occupied)
        res = self.fusion.evaluate(
            ai_prediction=ai_res,
            ttc=1.3,
            ttc_risk="CRITICAL",
            deterministic_clearance=40.0,
            lateral_maneuver_allowed=True,
            target_lane_safe=False,
            sim_time=6.5,
        )

        self.assertEqual(res.decision_reason, "BLOCKED_LANE_OVERRIDE")
        self.assertEqual(res.recommended_action, "EMERGENCY_BRAKE")

    def test_09_fused_result_is_deterministic(self):
        """Test 9: Repeated evaluations with identical arguments yield identical outputs."""
        ai_res = AIPredictionResult(
            sender_id="C-01",
            prediction_available=True,
            ai_risk_state="PREDICTED_CONFLICT",
            predicted_conflict=True,
            predicted_min_distance=22.0,
        )

        args = dict(
            ai_prediction=ai_res,
            ttc=2.5,
            ttc_risk="WARNING",
            deterministic_clearance=70.0,
            lateral_maneuver_allowed=True,
            target_lane_safe=True,
            sim_time=4.0,
        )

        res1 = self.fusion.evaluate(**args)
        res2 = self.fusion.evaluate(**args)

        self.assertEqual(res1.fused_risk_state, res2.fused_risk_state)
        self.assertEqual(res1.decision_reason, res2.decision_reason)
        self.assertEqual(res1.recommended_action, res2.recommended_action)

    def test_10_ai_prediction_does_not_create_unsafe_maneuver(self):
        """Test 10: AI alone cannot trigger lateral evasion when physical clearance is large."""
        ai_res = AIPredictionResult(
            sender_id="C-01",
            prediction_available=True,
            ai_risk_state="PREDICTED_CONFLICT",
            predicted_conflict=True,
            predicted_min_distance=40.0,
        )

        # 200px clearance, safe speed, TTC = inf
        res = self.fusion.evaluate(
            ai_prediction=ai_res,
            ttc=float("inf"),
            ttc_risk="NONE",
            deterministic_clearance=200.0,
            lateral_maneuver_allowed=True,
            target_lane_safe=True,
            sim_time=2.0,
        )

        self.assertNotEqual(res.recommended_action, "EVASIVE_MANEUVER")
        self.assertEqual(res.recommended_action, "CONTINUE")

    def test_11_ai_vs_ttc_timing_measurement(self):
        """Test 11: Instrumentation accurately measures when AI reports conflict vs when TTC reaches critical."""
        ai_conflict = AIPredictionResult(
            sender_id="C-01",
            prediction_available=True,
            ai_risk_state="PREDICTED_CONFLICT",
            predicted_conflict=True,
            predicted_min_distance=20.0,
        )

        self.fusion.reset()

        # Step A: at t=5.0s, AI first detects conflict (TTC is still WARNING)
        self.fusion.evaluate(
            ai_prediction=ai_conflict,
            ttc=2.8,
            ttc_risk="WARNING",
            deterministic_clearance=80.0,
            lateral_maneuver_allowed=True,
            target_lane_safe=True,
            sim_time=5.0,
        )
        self.assertEqual(self.fusion.first_ai_conflict_time, 5.0)
        self.assertIsNone(self.fusion.first_ttc_critical_time)

        # Step B: at t=5.8s, TTC reaches CRITICAL
        self.fusion.evaluate(
            ai_prediction=ai_conflict,
            ttc=1.4,
            ttc_risk="CRITICAL",
            deterministic_clearance=35.0,
            lateral_maneuver_allowed=True,
            target_lane_safe=True,
            sim_time=5.8,
        )
        self.assertEqual(self.fusion.first_ttc_critical_time, 5.8)
        self.assertIsNotNone(self.fusion.ai_lead_time)
        self.assertAlmostEqual(self.fusion.ai_lead_time, 0.8, places=2)

    def test_12_existing_v2v_behavior_regression(self):
        """Test 12: Integrated simulation with safety fusion preserves collision-free evasive maneuver."""
        random.seed(42)
        sig = SmartTrafficSignal("INT-TEST", 560, 415)
        predictor = IntersectionTrajectoryPredictor(base_dir=".")
        fusion = IntersectionSafetyFusion()
        tm = TrafficManager(enable_v2v_hazard=True, ai_predictor=predictor, safety_fusion=fusion)
        v2i = V2IManager(v2i_range=400.0, latency=0.20)
        v2v = IntersectionV2VManager(v2v_range=200.0, latency=0.020)

        amb = AmbulanceVehicle("AMB-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=650.0, speed=95.0)
        amb.is_authorized = True
        tm.ambulance = amb

        c01 = CivilianVehicle("C-01", "SOUTH", x=SOUTH_PRIMARY_LANE_X, y=550.0, speed=15.0)
        c01.set_hazard("DECELERATING", target_speed=15.0)
        tm.vehicles.append(c01)

        sim_time = 0.0
        for _ in range(250):
            sim_time += 1.0 / 60.0
            sig.update(1.0 / 60.0)
            tm.update(1.0 / 60.0, sig, sim_time=sim_time, v2i_channel=v2i, rsu_pos=(560, 415), v2v_inter_manager=v2v)

        # Vehicle must have completed evasive maneuver safely
        self.assertTrue(tm.ambulance.v2v_evasion_complete)
        self.assertAlmostEqual(tm.ambulance.x, SOUTH_OVERTAKE_LANE_X, delta=1.0)
        # Safety fusion was active and captured results
        self.assertIsNotNone(tm.ambulance.latest_safety_fusion)


if __name__ == "__main__":
    unittest.main()
