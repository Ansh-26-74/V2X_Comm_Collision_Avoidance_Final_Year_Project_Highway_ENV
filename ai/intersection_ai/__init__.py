"""Intersection AI Subsystem for Urban V2X Environment (Phase 13D).

Provides PyTorch-based vehicle trajectory prediction, decoupled geometric conflict
analysis, and deterministic safety fusion specifically for the urban smart intersection scenario.
Isolated from the highway V2V AI subsystem.
"""

def __getattr__(name: str):
    if name == "IntersectionTrajectoryMLP":
        from ai.intersection_ai.model import IntersectionTrajectoryMLP
        return IntersectionTrajectoryMLP
    elif name in ("IntersectionTrajectoryPredictor", "AIPredictionResult", "TrajectoryWaypoint", "GeometricConflictAnalyzer"):
        import ai.intersection_ai.predictor as p
        return getattr(p, name)
    elif name in ("IntersectionSafetyFusion", "IntersectionSafetyFusionResult"):
        import ai.intersection_ai.safety_fusion as sf
        return getattr(sf, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "IntersectionTrajectoryMLP",
    "IntersectionTrajectoryPredictor",
    "AIPredictionResult",
    "TrajectoryWaypoint",
    "GeometricConflictAnalyzer",
    "IntersectionSafetyFusion",
    "IntersectionSafetyFusionResult",
]
