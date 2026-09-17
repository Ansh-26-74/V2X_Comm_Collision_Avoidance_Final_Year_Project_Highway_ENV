"""Phase 13D Step 2: AI Inference Engine for Intersection Vehicle Trajectory Prediction.

Loads the trained PyTorch IntersectionTrajectoryMLP and feature scaler,
maintains a rolling history of the latest 5 V2V telemetry samples per vehicle,
predicts a 1.50s future trajectory (6 waypoints), and applies a decoupled
deterministic geometric conflict analysis against the emergency corridor.
"""

from collections import deque
from dataclasses import dataclass, field
import math
import os
import pickle
import time
from typing import Any, Sequence
import numpy as np

# Discrete prediction horizon offsets in seconds
FORECAST_HORIZONS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
HISTORY_LENGTH = 5


@dataclass
class TrajectoryWaypoint:
    """A single predicted future position waypoint."""
    horizon_offset_s: float
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.horizon_offset_s, self.x, self.y)


@dataclass
class AIPredictionResult:
    """Encapsulates the output of trajectory prediction and decoupled geometric analysis."""
    sender_id: str
    prediction_available: bool = False
    horizon_seconds: float = 1.50
    waypoints: list[TrajectoryWaypoint] = field(default_factory=list)
    ai_risk_state: str = "INSUFFICIENT_DATA"  # "INSUFFICIENT_DATA", "PREDICTED_SAFE", "PREDICTED_CONFLICT"
    predicted_conflict: bool = False
    predicted_min_distance: float = float("inf")
    inference_time_ms: float = 0.0
    status_message: str = "INITIALIZING"

    def get_waypoint_coords(self) -> list[tuple[float, float]]:
        """Return list of (x, y) coordinates for all waypoints."""
        return [(wp.x, wp.y) for wp in self.waypoints]


@dataclass
class TelemetrySample:
    """Validated internal telemetry sample."""
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    heading: float
    hazard_code: float
    timestamp: float
    accel: float = 0.0


def parse_hazard_status(hazard: str | int | float) -> float:
    """Normalize hazard status string to scalar code: 0.0=NORMAL, 0.5=DECELERATING, 1.0=HAZARD/STOPPED."""
    if isinstance(hazard, (int, float)):
        return float(hazard)
    h_str = str(hazard).upper()
    if "DECEL" in h_str:
        return 0.5
    elif "STOP" in h_str or "HAZARD" in h_str or "CRITICAL" in h_str:
        return 1.0
    return 0.0


class GeometricConflictAnalyzer:
    """Decoupled deterministic geometric conflict evaluator.
    
    Determines whether predicted lead vehicle waypoints penetrate the ambulance's
    forward safety corridor. Operates purely on kinematic geometry, independent of ML.
    """

    def __init__(
        self,
        corridor_half_width: float = 24.0,  # pixels (lane width is ~36px)
        safe_bumper_buffer: float = 75.0,   # pixels minimum forward clearance (~7.5m)
    ):
        self.corridor_half_width = corridor_half_width
        self.safe_bumper_buffer = safe_bumper_buffer

    def evaluate(
        self,
        waypoints: list[TrajectoryWaypoint],
        amb_x: float,
        amb_y: float,
        amb_speed: float,
        amb_vy: float = -1.0,  # Northbound
    ) -> tuple[bool, float, str]:
        """Evaluate predicted waypoints against the projected ambulance corridor.

        Returns (predicted_conflict, min_clearance_dist, risk_state).
        """
        if not waypoints:
            return False, float("inf"), "INSUFFICIENT_DATA"

        min_dist = float("inf")
        conflict_detected = False

        for wp in waypoints:
            dt = wp.horizon_offset_s
            # Projected position of ambulance at future time dt
            # (Northbound movement: y decreases by amb_speed * dt)
            amb_future_y = amb_y + (amb_vy * amb_speed * dt)
            amb_future_x = amb_x

            # Lateral offset from ambulance corridor
            lat_offset = abs(wp.x - amb_future_x)

            if lat_offset <= self.corridor_half_width:
                # Longitudinal separation (Northbound: lead vehicle has smaller Y if ahead)
                # gap = amb_future_y - lead_future_y. Positive means lead is ahead of ambulance.
                longitudinal_gap = amb_future_y - wp.y

                # Spatial distance between centers
                spatial_dist = math.hypot(wp.x - amb_future_x, wp.y - amb_future_y)
                if spatial_dist < min_dist:
                    min_dist = spatial_dist

                # Conflict criteria:
                # Lead vehicle is ahead within the safety buffer, or ambulance has overtaken/penetrated
                if longitudinal_gap <= self.safe_bumper_buffer:
                    conflict_detected = True

        risk_state = "PREDICTED_CONFLICT" if conflict_detected else "PREDICTED_SAFE"
        return conflict_detected, min_dist, risk_state


class IntersectionTrajectoryPredictor:
    """PyTorch-based Trajectory Predictor for the Urban Smart Intersection.

    Maintains a 5-step rolling telemetry history per vehicle, converts observations
    to normalized 40-feature inputs, performs batch inference, and produces 6 future waypoints.
    """

    def __init__(
        self,
        base_dir: str = ".",
        model_path: str | None = None,
        scaler_path: str | None = None,
        device: str = "cpu",
    ):
        import torch
        from ai.intersection_ai.model import IntersectionTrajectoryMLP

        self.device = torch.device(device)
        self.history_buffers: dict[str, deque[TelemetrySample]] = {}
        self.conflict_analyzer = GeometricConflictAnalyzer()
        self.is_ready: bool = False
        self.status: str = "INITIALIZING"

        # Resolve paths
        models_dir = os.path.join(base_dir, "models", "intersection")
        if model_path is None:
            model_path = os.path.join(models_dir, "trajectory_model.pth")
        if scaler_path is None:
            scaler_path = os.path.join(models_dir, "feature_scaler.pkl")

        self.model_path = model_path
        self.scaler_path = scaler_path

        # Load scaler with fast unpickler to bypass heavy scikit-learn/scipy imports
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Feature scaler not found at: {scaler_path}")

        class _FastScalerUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if "sklearn" in module:
                    class _FastStandardScaler:
                        mean_: np.ndarray
                        scale_: np.ndarray

                        def transform(self, X: np.ndarray) -> np.ndarray:
                            return (X - self.mean_) / self.scale_
                    return _FastStandardScaler
                return super().find_class(module, name)

        try:
            with open(scaler_path, "rb") as f:
                self.scaler = _FastScalerUnpickler(f).load()
        except Exception:
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        # Load PyTorch model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PyTorch model weights not found at: {model_path}")

        self.model = IntersectionTrajectoryMLP(input_dim=40, hidden_dim=64, output_dim=12).to(self.device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.model.eval()

        self.is_ready = True
        self.status = "ONLINE"

    def reset(self, sender_id: str | None = None) -> None:
        """Clear telemetry history buffers."""
        if sender_id is not None:
            if sender_id in self.history_buffers:
                self.history_buffers[sender_id].clear()
        else:
            self.history_buffers.clear()

    def add_telemetry(self, msg: Any) -> bool:
        """Validate and append a V2V telemetry sample to the vehicle's rolling history.

        Accepts IntersectionV2VMessage or dict with keys:
        'sender_id', 'x', 'y', 'vx', 'vy', 'speed', 'heading', 'hazard_status', 'timestamp'.
        
        Returns True if accepted, False if rejected (malformed / non-finite / duplicate).
        """
        # Extract fields
        try:
            if isinstance(msg, dict):
                sender_id = str(msg["sender_id"])
                x = float(msg["x"])
                y = float(msg["y"])
                vx = float(msg.get("vx", 0.0))
                vy = float(msg.get("vy", -1.0))
                speed = float(msg["speed"])
                heading = float(msg.get("heading", -math.pi / 2.0))
                hazard = msg.get("hazard_status", "NORMAL")
                timestamp = float(msg["timestamp"])
            else:
                sender_id = str(getattr(msg, "sender_id"))
                x = float(getattr(msg, "x"))
                y = float(getattr(msg, "y"))
                vx = float(getattr(msg, "vx", 0.0))
                vy = float(getattr(msg, "vy", -1.0))
                speed = float(getattr(msg, "speed"))
                heading = float(getattr(msg, "heading", -math.pi / 2.0))
                hazard = getattr(msg, "hazard_status", "NORMAL")
                timestamp = float(getattr(msg, "timestamp"))
        except (AttributeError, KeyError, TypeError, ValueError):
            return False

        # Reject non-finite values (NaN / inf)
        vals = [x, y, vx, vy, speed, heading, timestamp]
        if any(math.isnan(v) or math.isinf(v) for v in vals):
            return False

        if speed < 0.0:
            speed = 0.0

        hazard_code = parse_hazard_status(hazard)

        # Retrieve buffer
        if sender_id not in self.history_buffers:
            self.history_buffers[sender_id] = deque(maxlen=HISTORY_LENGTH)
        buf = self.history_buffers[sender_id]

        # Monotonic timestamp check (reject duplicate or out-of-order samples)
        if len(buf) > 0 and timestamp <= buf[-1].timestamp:
            return False

        # Calculate acceleration relative to previous sample
        accel = 0.0
        if len(buf) > 0:
            dt = timestamp - buf[-1].timestamp
            if dt > 1e-4:
                accel = (speed - buf[-1].speed) / dt

        sample = TelemetrySample(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            speed=speed,
            heading=heading,
            hazard_code=hazard_code,
            timestamp=timestamp,
            accel=accel,
        )
        buf.append(sample)
        return True

    def get_history_count(self, sender_id: str) -> int:
        """Return number of buffered samples for the given vehicle."""
        if sender_id not in self.history_buffers:
            return 0
        return len(self.history_buffers[sender_id])

    def build_features(self, sender_id: str) -> np.ndarray | None:
        """Construct the exact 40-feature vector matching the training representation.

        Returns np.ndarray of shape (1, 40) or None if history < 5.
        """
        buf = self.history_buffers.get(sender_id)
        if buf is None or len(buf) < HISTORY_LENGTH:
            return None

        curr = buf[-1]
        curr_x = curr.x
        curr_y = curr.y

        feats = []
        for sample in buf:
            dx = sample.x - curr_x
            dy = sample.y - curr_y
            feats.extend([
                dx,
                dy,
                sample.vx,
                sample.vy,
                sample.speed / 100.0,       # Normalized speed
                sample.heading / math.pi,   # Normalized heading
                sample.accel / 100.0,       # Normalized acceleration
                sample.hazard_code,
            ])

        return np.array([feats], dtype=np.float32)

    def predict(
        self,
        sender_id: str,
        ambulance_pos: tuple[float, float] | None = None,
        ambulance_speed: float = 95.0,
    ) -> AIPredictionResult:
        """Run trajectory inference for sender_id and evaluate deterministic geometric corridor risk.

        Parameters
        ----------
        sender_id : str
            Identifier of tracked lead vehicle (e.g. "C-01").
        ambulance_pos : tuple[float, float], optional
            Current (x, y) coordinates of the approaching ambulance.
        ambulance_speed : float
            Current speed of the ambulance (px/s).

        Returns
        -------
        AIPredictionResult
            Structured prediction result with 6 future waypoints and geometric conflict status.
        """
        start_time = time.perf_counter()

        # Check model readiness
        if not self.is_ready:
            return AIPredictionResult(
                sender_id=sender_id,
                prediction_available=False,
                ai_risk_state="INSUFFICIENT_DATA",
                status_message="MODEL_NOT_READY",
            )

        # Check history length
        feats = self.build_features(sender_id)
        if feats is None:
            return AIPredictionResult(
                sender_id=sender_id,
                prediction_available=False,
                ai_risk_state="INSUFFICIENT_DATA",
                status_message=f"BUFFER_NOT_FULL ({self.get_history_count(sender_id)}/{HISTORY_LENGTH})",
            )

        try:
            import torch
            # 1. Feature normalization
            feats_scaled = self.scaler.transform(feats)

            # 2. PyTorch evaluation
            x_tensor = torch.tensor(feats_scaled, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out_tensor = self.model(x_tensor)
            pred_raw = out_tensor.cpu().numpy()[0]

            # 3. Convert 12 relative offsets to 6 absolute waypoints
            curr = self.history_buffers[sender_id][-1]
            curr_x = curr.x
            curr_y = curr.y

            waypoints: list[TrajectoryWaypoint] = []
            for i, h_offset in enumerate(FORECAST_HORIZONS):
                dx = float(pred_raw[2 * i])
                dy = float(pred_raw[2 * i + 1])
                wp = TrajectoryWaypoint(
                    horizon_offset_s=h_offset,
                    x=curr_x + dx,
                    y=curr_y + dy,
                )
                waypoints.append(wp)

            inference_ms = (time.perf_counter() - start_time) * 1000.0

            # 4. Decoupled Deterministic Geometric Conflict Analysis
            conflict = False
            min_dist = float("inf")
            risk_state = "PREDICTED_SAFE"

            if ambulance_pos is not None:
                amb_x, amb_y = ambulance_pos
                conflict, min_dist, risk_state = self.conflict_analyzer.evaluate(
                    waypoints=waypoints,
                    amb_x=amb_x,
                    amb_y=amb_y,
                    amb_speed=ambulance_speed,
                )

            return AIPredictionResult(
                sender_id=sender_id,
                prediction_available=True,
                horizon_seconds=1.50,
                waypoints=waypoints,
                ai_risk_state=risk_state,
                predicted_conflict=conflict,
                predicted_min_distance=min_dist,
                inference_time_ms=inference_ms,
                status_message="OK",
            )

        except Exception as e:
            # Safe structured failure fallback: never crash calling system
            return AIPredictionResult(
                sender_id=sender_id,
                prediction_available=False,
                ai_risk_state="INSUFFICIENT_DATA",
                status_message=f"INFERENCE_ERROR: {str(e)}",
            )
