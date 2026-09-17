"""Phase 13D Step 4: Deterministic Safety Fusion Engine for Urban Intersection.

Fuses advisory AI trajectory forecasts with deterministic physical TTC, bumper clearance,
and roadway geometric safety constraints.

Safety Hierarchy:
1. Deterministic TTC / physical clearance is AUTHORITATIVE.
2. AI Trajectory Prediction is ADVISORY.
3. AI conflict cannot bypass conflict-zone rules, lane occupancy, or braking gates.
4. If AI is unavailable or produces an error, fusion falls back 100% to deterministic TTC.
"""

from dataclasses import dataclass
import math
from typing import Any

from ai.intersection_ai.predictor import AIPredictionResult


@dataclass
class IntersectionSafetyFusionResult:
    """Structured output from the deterministic safety fusion evaluation."""
    fused_risk_state: str          # "SAFE", "ADVISORY_MONITORING", "ELEVATED_WARNING", "CRITICAL"
    decision_reason: str           # Explicit rationale string
    recommended_action: str        # "CONTINUE", "PREPARE_EVASION", "EVASIVE_MANEUVER", "EMERGENCY_BRAKE"
    ai_risk_state: str             # "INSUFFICIENT_DATA", "PREDICTED_SAFE", "PREDICTED_CONFLICT"
    ai_conflict: bool              # True if AI predicts corridor conflict
    ttc: float                     # Instantaneous kinematic TTC (s)
    ttc_risk_state: str            # "NONE", "SAFE", "WARNING", "CRITICAL"
    predicted_min_distance: float  # Forecasted minimum clearance from AI (px)
    deterministic_clearance: float # Current physical bumper gap (px)
    lateral_maneuver_allowed: bool # True if outside conflict zone and authorized
    target_lane_safe: bool         # True if adjacent lane has clearance ahead and behind
    timestamp: float               # Simulation timestamp (s)


class IntersectionSafetyFusion:
    """Deterministic Safety Fusion Engine for the urban intersection scenario.
    
    Evaluates advisory AI trajectory forecasts alongside authoritative physical TTC,
    bumper gaps, lane occupancy, and conflict-zone boundaries.
    """

    def __init__(
        self,
        critical_ttc_threshold: float = 2.0,   # seconds
        warning_ttc_threshold: float = 4.0,    # seconds
        critical_gap_threshold: float = 25.0,  # pixels
    ):
        self.critical_ttc_threshold = critical_ttc_threshold
        self.warning_ttc_threshold = warning_ttc_threshold
        self.critical_gap_threshold = critical_gap_threshold

        # Instrumentation for measuring AI vs TTC lead time
        self.first_ai_conflict_time: float | None = None
        self.first_ttc_critical_time: float | None = None
        self.ai_lead_time: float | None = None

    def reset(self) -> None:
        """Reset timing metrics and internal state."""
        self.first_ai_conflict_time = None
        self.first_ttc_critical_time = None
        self.ai_lead_time = None

    def evaluate(
        self,
        ai_prediction: AIPredictionResult | None,
        ttc: float,
        ttc_risk: str,
        deterministic_clearance: float,
        lateral_maneuver_allowed: bool,
        target_lane_safe: bool,
        sim_time: float = 0.0,
        in_intersection: bool = False,
    ) -> IntersectionSafetyFusionResult:
        """Deterministically fuse AI advisory prediction and physical TTC constraints.

        Parameters
        ----------
        ai_prediction : AIPredictionResult or None
            Latest advisory trajectory prediction from the neural network.
        ttc : float
            Instantaneous kinematic Time-To-Collision (seconds).
        ttc_risk : str
            Instantaneous kinematic risk state ("NONE", "SAFE", "WARNING", "CRITICAL").
        deterministic_clearance : float
            Current physical bumper gap to lead vehicle (pixels).
        lateral_maneuver_allowed : bool
            True if vehicle is outside the central conflict zone and maneuver is physically permitted.
        target_lane_safe : bool
            True if prospective target lane is clear of vehicles ahead and behind.
        sim_time : float
            Current simulation timestamp.
        in_intersection : bool
            True if vehicle is already inside the central intersection conflict zone.

        Returns
        -------
        IntersectionSafetyFusionResult
            Authoritative fused risk state, decision reason, and recommended action.
        """
        # Extract AI parameters
        ai_available = ai_prediction is not None and getattr(ai_prediction, "prediction_available", False)
        ai_risk = getattr(ai_prediction, "ai_risk_state", "INSUFFICIENT_DATA") if ai_prediction else "INSUFFICIENT_DATA"
        ai_conflict = getattr(ai_prediction, "predicted_conflict", False) if ai_prediction else False
        pred_min_dist = getattr(ai_prediction, "predicted_min_distance", float("inf")) if ai_prediction else float("inf")

        # ── Instrumentation: Measure AI vs. TTC Timing ───────────────────────
        if ai_conflict and self.first_ai_conflict_time is None and sim_time > 0.0:
            self.first_ai_conflict_time = round(sim_time, 3)

        if ttc_risk == "CRITICAL" and self.first_ttc_critical_time is None and sim_time > 0.0:
            self.first_ttc_critical_time = round(sim_time, 3)
            if self.first_ai_conflict_time is not None:
                self.ai_lead_time = round(self.first_ttc_critical_time - self.first_ai_conflict_time, 3)

        # ── Authoritative Hierarchy Level 1: Deterministic CRITICAL ──────────
        # If physical TTC < critical threshold OR physical gap <= critical bumper gap,
        # threat is physically imminent. Deterministic safety triggers action regardless of AI.
        is_physically_critical = (
            ttc_risk == "CRITICAL"
            or ttc < self.critical_ttc_threshold
            or deterministic_clearance <= self.critical_gap_threshold
        )

        if is_physically_critical:
            fused_risk = "CRITICAL"

            # Check geometric constraints for action selection
            if not lateral_maneuver_allowed or in_intersection:
                # Rule 6: AI cannot override conflict zone restrictions
                reason = "CONFLICT_ZONE_OVERRIDE"
                action = "CONTINUE" if in_intersection else "EMERGENCY_BRAKE"
            elif not target_lane_safe:
                # Rule 6: Target lane blocked -> emergency braking fallback
                reason = "BLOCKED_LANE_OVERRIDE"
                action = "EMERGENCY_BRAKE"
            else:
                # Target lane safe and outside conflict zone -> safe evasion
                reason = "TTC_CRITICAL"
                action = "EVASIVE_MANEUVER"

            return IntersectionSafetyFusionResult(
                fused_risk_state=fused_risk,
                decision_reason=reason,
                recommended_action=action,
                ai_risk_state=ai_risk,
                ai_conflict=ai_conflict,
                ttc=ttc,
                ttc_risk_state=ttc_risk,
                predicted_min_distance=pred_min_dist,
                deterministic_clearance=deterministic_clearance,
                lateral_maneuver_allowed=lateral_maneuver_allowed,
                target_lane_safe=target_lane_safe,
                timestamp=sim_time,
            )

        # ── Hierarchy Level 2: AI Predicted Conflict with TTC Evidence ────────
        if ai_available and ai_conflict:
            # Case 2A: AI predicts conflict AND physical TTC shows elevated risk (WARNING: 2.0s - 4.0s)
            if ttc_risk == "WARNING" or ttc <= self.warning_ttc_threshold:
                fused_risk = "ELEVATED_WARNING"
                reason = "AI_CONFLICT_WITH_TTC_WARNING"
                action = "PREPARE_EVASION"
            else:
                # Case 2B: AI predicts conflict BUT physical TTC is SAFE (> 4.0s) and clearance is large
                # Rule 4 & 5: AI predicted conflict alone does NOT force an evasive maneuver.
                fused_risk = "ADVISORY_MONITORING"
                reason = "AI_CONFLICT_BUT_TTC_SAFE"
                action = "CONTINUE"

            return IntersectionSafetyFusionResult(
                fused_risk_state=fused_risk,
                decision_reason=reason,
                recommended_action=action,
                ai_risk_state=ai_risk,
                ai_conflict=ai_conflict,
                ttc=ttc,
                ttc_risk_state=ttc_risk,
                predicted_min_distance=pred_min_dist,
                deterministic_clearance=deterministic_clearance,
                lateral_maneuver_allowed=lateral_maneuver_allowed,
                target_lane_safe=target_lane_safe,
                timestamp=sim_time,
            )

        # ── Hierarchy Level 3: AI Unavailable / Error Fallback ────────────────
        if not ai_available or "ERROR" in getattr(ai_prediction, "status_message", ""):
            # Rule 5: Completely fall back to existing deterministic TTC behavior
            reason = "AI_UNAVAILABLE_DETERMINISTIC_FALLBACK"
            fused_risk = ttc_risk if ttc_risk in ("WARNING", "SAFE") else "SAFE"
            action = "CONTINUE"

            return IntersectionSafetyFusionResult(
                fused_risk_state=fused_risk,
                decision_reason=reason,
                recommended_action=action,
                ai_risk_state=ai_risk,
                ai_conflict=False,
                ttc=ttc,
                ttc_risk_state=ttc_risk,
                predicted_min_distance=pred_min_dist,
                deterministic_clearance=deterministic_clearance,
                lateral_maneuver_allowed=lateral_maneuver_allowed,
                target_lane_safe=target_lane_safe,
                timestamp=sim_time,
            )

        # ── Hierarchy Level 4: Clear Roadway / All Systems Safe ───────────────
        return IntersectionSafetyFusionResult(
            fused_risk_state="SAFE",
            decision_reason="SAFE",
            recommended_action="CONTINUE",
            ai_risk_state=ai_risk,
            ai_conflict=False,
            ttc=ttc,
            ttc_risk_state=ttc_risk,
            predicted_min_distance=pred_min_dist,
            deterministic_clearance=deterministic_clearance,
            lateral_maneuver_allowed=lateral_maneuver_allowed,
            target_lane_safe=target_lane_safe,
            timestamp=sim_time,
        )
