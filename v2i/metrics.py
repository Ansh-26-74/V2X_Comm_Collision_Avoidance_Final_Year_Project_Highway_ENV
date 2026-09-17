"""Phase 10: Quantitative V2I Benefit Measurement & Benchmark System.

Provides:
- ExperimentMode: BASELINE_NO_V2I vs V2I_ENABLED
- TrialStatus: COMPLETED, TIMEOUT, ERROR
- TrialConfig: Configuration for deterministic reproducible trials
- TrialResult: Measured metrics for a single simulation run
- MetricsCollector: Real-time observer capturing timestamps, state transitions, stops, and wait times
- SimulationRunner: Headless deterministic trial execution harness
- ExperimentSuite: Paired comparison runner (Baseline vs V2I across N trials)
- ComparisonReport: Summary statistics, paired differences, percentage reductions, and JSON/CSV export
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import csv
import json
import logging
import math
import random
from typing import Any

from v2i.smart_signal import SmartTrafficSignal, CyclePhase, SignalState
from v2i.traffic_manager import TrafficManager, AmbulanceState
from v2i.v2i_manager import V2IManager
import v2i.v2i_renderer as renderer

logger = logging.getLogger(__name__)


class ExperimentMode(str, Enum):
    """Execution mode for comparative evaluation."""
    BASELINE_NO_V2I = "BASELINE_NO_V2I"
    V2I_ENABLED = "V2I_ENABLED"


class TrialStatus(str, Enum):
    """Outcome status of an individual trial."""
    COMPLETED = "COMPLETED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


@dataclass
class TrialConfig:
    """Deterministic configuration parameters for a single trial."""
    trial_id: int
    seed: int
    ambulance_spawn_time: float = 8.5
    initial_phase: CyclePhase = CyclePhase.NS_GREEN_EW_RED
    initial_phase_timer: float = 0.0
    timeout: float = 60.0
    description: str = ""


@dataclass
class TrialResult:
    """Recorded metrics for an individual simulation trial."""
    mode: str
    trial_id: int
    seed: int
    status: str
    measurement_start: float
    measurement_end: float
    total_travel_time: float | None
    signal_wait_time: float
    traffic_wait_time: float
    intersection_clearance_time: float | None
    total_stops: int
    signal_stops: int
    traffic_stops: int
    stop_line_crossing_time: float | None = None
    emergency_request_time: float | None = None
    rsu_reception_time: float | None = None
    preemption_activation_time: float | None = None
    preemption_delay: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        d = asdict(self)
        # Round floating point metrics for clean presentation
        for k in (
            "measurement_start", "measurement_end", "total_travel_time",
            "signal_wait_time", "traffic_wait_time", "intersection_clearance_time",
            "stop_line_crossing_time", "emergency_request_time",
            "rsu_reception_time", "preemption_activation_time", "preemption_delay",
        ):
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 3)
        return d


class MetricsCollector:
    """Frame-by-frame observer collecting authoritative simulation metrics."""

    def __init__(self, mode: ExperimentMode, trial_config: TrialConfig):
        self.mode = mode
        self.config = trial_config

        # Lifecycle timestamps
        self.spawn_time: float | None = None
        self.clearance_time: float | None = None
        self.stop_line_crossing_time: float | None = None

        # Durations
        self.signal_wait_time: float = 0.0
        self.traffic_wait_time: float = 0.0

        # Stop tracking
        self.total_stops: int = 0
        self.signal_stops: int = 0
        self.traffic_stops: int = 0
        self._was_stopped: bool = False

        # V2I communication timestamps
        self.emergency_request_time: float | None = None
        self.rsu_reception_time: float | None = None
        self.preemption_activation_time: float | None = None

        # Observer tracking
        self._last_transmitted_count: int = 0
        self._last_delivered_count: int = 0
        self._last_phase: CyclePhase | None = None

    def step(
        self,
        sim_time: float,
        dt: float,
        signal_controller: SmartTrafficSignal,
        traffic_manager: TrafficManager,
        v2i_channel: V2IManager | None = None,
    ) -> None:
        """Inspect simulation components each tick and record genuine state changes."""
        amb = traffic_manager.ambulance

        # 1. Capture measurement start: ambulance spawn time
        if amb is not None and self.spawn_time is None:
            self.spawn_time = sim_time

        if amb is None:
            return

        # 2. Capture stop-line crossing (South stop line is at cy + stop_line_dist)
        # Moving North (-Y): when front bumper y <= cy + stop_line_dist
        if self.stop_line_crossing_time is None and amb.past_stop_line:
            self.stop_line_crossing_time = sim_time

        # 3. Capture full-body intersection clearance
        if self.clearance_time is None and amb.cleared:
            self.clearance_time = sim_time

        # 4. If already cleared, do not accumulate further stops or waits
        if self.clearance_time is not None:
            return

        # 5. Measure waiting time by cause
        amb_state = amb.ambulance_state
        is_stopped = (amb.speed <= 0.05)

        if is_stopped:
            if amb_state == AmbulanceState.WAITING_AT_RED:
                self.signal_wait_time += dt
            elif amb_state in (AmbulanceState.STOPPED, AmbulanceState.DECELERATING):
                self.traffic_wait_time += dt

        # 6. Deduplicated stop counting (transition into stopped condition)
        if is_stopped and not self._was_stopped:
            self._was_stopped = True
            self.total_stops += 1
            if amb_state == AmbulanceState.WAITING_AT_RED:
                self.signal_stops += 1
            else:
                self.traffic_stops += 1
        elif not is_stopped and self._was_stopped:
            self._was_stopped = False

        # 7. V2I Specific Timestamps (V2I_ENABLED mode only)
        if self.mode == ExperimentMode.V2I_ENABLED and v2i_channel is not None:
            # First transmission
            if self.emergency_request_time is None and v2i_channel.total_transmitted > 0:
                self.emergency_request_time = sim_time

            # First delivery to RSU
            if self.rsu_reception_time is None and v2i_channel.total_delivered > 0:
                self.rsu_reception_time = sim_time

            # Preemption activation: transition to PREEMPTION_YELLOW, PREEMPTION_ALL_RED, or EMERGENCY_SOUTH_GREEN
            curr_phase = signal_controller.current_phase
            if self.preemption_activation_time is None:
                if (
                    signal_controller.preemption_requested
                    or curr_phase in (
                        CyclePhase.PREEMPTION_YELLOW,
                        CyclePhase.PREEMPTION_ALL_RED,
                        CyclePhase.EMERGENCY_SOUTH_GREEN,
                    )
                ):
                    self.preemption_activation_time = sim_time

        self._last_phase = signal_controller.current_phase

    def finalize(self, sim_time: float) -> TrialResult:
        """Construct the finalized TrialResult from collected metrics."""
        status = TrialStatus.COMPLETED if self.clearance_time is not None else TrialStatus.TIMEOUT
        start_t = self.spawn_time if self.spawn_time is not None else self.config.ambulance_spawn_time

        if status == TrialStatus.COMPLETED:
            total_travel = self.clearance_time - start_t
            if self.stop_line_crossing_time is not None:
                clearance_duration = self.clearance_time - self.stop_line_crossing_time
            else:
                clearance_duration = None
            end_t = self.clearance_time
        else:
            total_travel = None
            clearance_duration = None
            end_t = sim_time

        preemption_delay = None
        if self.emergency_request_time is not None and self.preemption_activation_time is not None:
            preemption_delay = max(0.0, self.preemption_activation_time - self.emergency_request_time)

        return TrialResult(
            mode=self.mode.value,
            trial_id=self.config.trial_id,
            seed=self.config.seed,
            status=status.value,
            measurement_start=start_t,
            measurement_end=end_t,
            total_travel_time=total_travel,
            signal_wait_time=self.signal_wait_time,
            traffic_wait_time=self.traffic_wait_time,
            intersection_clearance_time=clearance_duration,
            total_stops=self.total_stops,
            signal_stops=self.signal_stops,
            traffic_stops=self.traffic_stops,
            stop_line_crossing_time=self.stop_line_crossing_time,
            emergency_request_time=self.emergency_request_time,
            rsu_reception_time=self.rsu_reception_time,
            preemption_activation_time=self.preemption_activation_time,
            preemption_delay=preemption_delay,
        )


class SimulationRunner:
    """Executes a single headless simulation trial deterministically."""

    @staticmethod
    def run_trial(
        mode: ExperimentMode,
        config: TrialConfig,
        dt: float = 1.0 / 60.0,
    ) -> TrialResult:
        """Run a single simulation trial under exact controlled parameters."""
        # 1. Deterministic PRNG seeding
        random.seed(config.seed)

        # 2. Initialize Signal Controller
        signal_controller = SmartTrafficSignal()
        signal_controller.current_phase = config.initial_phase
        signal_controller.phase_timer = config.initial_phase_timer

        # 3. Initialize Traffic Manager
        traffic_manager = TrafficManager(
            center_x=renderer.CX,
            center_y=renderer.CY,
            road_width=renderer.ROAD_WIDTH,
            stop_line_dist=renderer.STOP_LINE_DIST,
            ambulance_spawn_time=config.ambulance_spawn_time,
        )

        # 4. Initialize V2I Channel (active for V2I_ENABLED, None for BASELINE_NO_V2I)
        if mode == ExperimentMode.V2I_ENABLED:
            v2i_channel = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=0.0)
            rsu_pos = renderer.RSU_POS
        else:
            v2i_channel = None
            rsu_pos = None

        # 5. Metrics Collector
        collector = MetricsCollector(mode=mode, trial_config=config)

        sim_time = 0.0
        max_steps = int(config.timeout / dt)

        for _ in range(max_steps):
            # Advance signals
            signal_controller.update(dt)
            sim_time += dt

            # Advance traffic
            traffic_manager.update(
                dt=dt,
                signal_controller=signal_controller,
                sim_time=sim_time,
                v2i_channel=v2i_channel,
                rsu_pos=rsu_pos,
            )

            # Preemption delivery & handling in V2I_ENABLED mode
            if mode == ExperimentMode.V2I_ENABLED and v2i_channel is not None:
                # Consume Phase 8 clearance
                if (
                    traffic_manager.ambulance_cleared_events
                    and signal_controller.current_phase == CyclePhase.EMERGENCY_SOUTH_GREEN
                ):
                    signal_controller.notify_ambulance_cleared("AMB-01")

                # Deliver in-flight packets
                delivered = v2i_channel.deliver_to_rsu(sim_time)
                for msg in delivered:
                    signal_controller.receive_emergency_request(msg, current_time=sim_time, auto_preempt=True)

                signal_controller.update_traffic_analysis(
                    civilian_vehicles=traffic_manager.vehicles,
                    ambulance=traffic_manager.ambulance,
                    current_time=sim_time,
                )
                v2i_channel.update_animations(sim_time)

            # Record metrics this tick
            collector.step(
                sim_time=sim_time,
                dt=dt,
                signal_controller=signal_controller,
                traffic_manager=traffic_manager,
                v2i_channel=v2i_channel,
            )

            # Early exit if ambulance has fully cleared
            if collector.clearance_time is not None:
                break

        return collector.finalize(sim_time)


class ExperimentSuite:
    """Manages paired benchmark trials between BASELINE_NO_V2I and V2I_ENABLED."""

    @staticmethod
    def generate_default_configs(num_trials: int = 10) -> list[TrialConfig]:
        """Generate N diverse, deterministic trial configurations."""
        configs: list[TrialConfig] = []
        # Systematic variation across signal cycle phases and arrival timings:
        # Normal cycle is 22s: NS Green (8s), NS Yellow (3s), EW Green (8s), EW Yellow (3s).
        spawn_offsets = [
            8.5,   # Mid EW Green (faces Red)
            9.5,   # Mid EW Green
            11.0,  # EW Green peak
            12.5,  # Late EW Green
            14.0,  # Late EW Green
            6.0,   # Late NS Green / entering Yellow
            7.5,   # NS Yellow onset
            16.0,  # EW Yellow onset
            17.5,  # Approaching NS Green recovery
            10.0,  # Mid EW Green with heavy East-West crossing traffic
        ]

        for i in range(num_trials):
            spawn_t = spawn_offsets[i % len(spawn_offsets)]
            seed = 1001 + i * 37
            cfg = TrialConfig(
                trial_id=i + 1,
                seed=seed,
                ambulance_spawn_time=spawn_t,
                timeout=65.0,
                description=f"Trial {i+1:02d} (Spawn: {spawn_t:.1f}s, Seed: {seed})",
            )
            configs.append(cfg)
        return configs

    @classmethod
    def run_paired_suite(
        cls,
        configs: list[TrialConfig] | None = None,
        num_trials: int = 10,
    ) -> tuple[list[TrialResult], list[TrialResult]]:
        """Run both baseline and V2I modes across all trial configurations.

        Returns (baseline_results, v2i_results).
        """
        if configs is None:
            configs = cls.generate_default_configs(num_trials)

        baseline_results: list[TrialResult] = []
        v2i_results: list[TrialResult] = []

        for cfg in configs:
            logger.info(f"[BENCHMARK] Executing {cfg.description} — BASELINE_NO_V2I...")
            res_base = SimulationRunner.run_trial(ExperimentMode.BASELINE_NO_V2I, cfg)
            baseline_results.append(res_base)

            logger.info(f"[BENCHMARK] Executing {cfg.description} — V2I_ENABLED...")
            res_v2i = SimulationRunner.run_trial(ExperimentMode.V2I_ENABLED, cfg)
            v2i_results.append(res_v2i)

        return baseline_results, v2i_results


class ComparisonReport:
    """Calculates statistical metrics and formats human/machine reports."""

    def __init__(self, baseline_results: list[TrialResult], v2i_results: list[TrialResult]):
        self.baseline_results = baseline_results
        self.v2i_results = v2i_results
        self.num_trials = len(baseline_results)

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float]:
        """Compute mean, median, min, and max for a numerical list."""
        if not values:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        s = sorted(values)
        n = len(s)
        mean_val = sum(s) / n
        if n % 2 == 1:
            median_val = s[n // 2]
        else:
            median_val = (s[n // 2 - 1] + s[n // 2]) / 2.0
        return {
            "mean": mean_val,
            "median": median_val,
            "min": s[0],
            "max": s[-1],
        }

    def compute_summary(self) -> dict[str, Any]:
        """Calculate complete statistical comparison between Baseline and V2I."""
        # Filter completed trials for travel time
        base_completed = [r for r in self.baseline_results if r.status == TrialStatus.COMPLETED.value]
        v2i_completed = [r for r in self.v2i_results if r.status == TrialStatus.COMPLETED.value]

        base_tt = [r.total_travel_time for r in base_completed if r.total_travel_time is not None]
        v2i_tt = [r.total_travel_time for r in v2i_completed if r.total_travel_time is not None]

        base_sw = [r.signal_wait_time for r in self.baseline_results]
        v2i_sw = [r.signal_wait_time for r in self.v2i_results]

        base_cl = [r.intersection_clearance_time for r in base_completed if r.intersection_clearance_time is not None]
        v2i_cl = [r.intersection_clearance_time for r in v2i_completed if r.intersection_clearance_time is not None]

        base_stops = [float(r.total_stops) for r in self.baseline_results]
        v2i_stops = [float(r.total_stops) for r in self.v2i_results]

        v2i_preempt_delays = [r.preemption_delay for r in v2i_completed if r.preemption_delay is not None]

        stats_base_tt = self._stats(base_tt)
        stats_v2i_tt = self._stats(v2i_tt)

        stats_base_sw = self._stats(base_sw)
        stats_v2i_sw = self._stats(v2i_sw)

        stats_base_cl = self._stats(base_cl)
        stats_v2i_cl = self._stats(v2i_cl)

        stats_base_stops = self._stats(base_stops)
        stats_v2i_stops = self._stats(v2i_stops)

        stats_v2i_delay = self._stats(v2i_preempt_delays)

        # Paired differences and percentage reductions
        diff_mean_tt = stats_base_tt["mean"] - stats_v2i_tt["mean"]
        pct_tt = (diff_mean_tt / stats_base_tt["mean"] * 100.0) if stats_base_tt["mean"] > 0 else 0.0

        diff_mean_sw = stats_base_sw["mean"] - stats_v2i_sw["mean"]
        pct_sw = (diff_mean_sw / stats_base_sw["mean"] * 100.0) if stats_base_sw["mean"] > 0 else 0.0

        diff_mean_stops = stats_base_stops["mean"] - stats_v2i_stops["mean"]
        pct_stops = (diff_mean_stops / stats_base_stops["mean"] * 100.0) if stats_base_stops["mean"] > 0 else 0.0

        return {
            "total_trials": self.num_trials,
            "baseline_completed": len(base_completed),
            "baseline_timeouts": self.num_trials - len(base_completed),
            "v2i_completed": len(v2i_completed),
            "v2i_timeouts": self.num_trials - len(v2i_completed),
            "travel_time": {
                "baseline": stats_base_tt,
                "v2i": stats_v2i_tt,
                "difference_mean": diff_mean_tt,
                "reduction_percent": pct_tt,
            },
            "signal_wait_time": {
                "baseline": stats_base_sw,
                "v2i": stats_v2i_sw,
                "difference_mean": diff_mean_sw,
                "reduction_percent": pct_sw,
            },
            "clearance_time": {
                "baseline": stats_base_cl,
                "v2i": stats_v2i_cl,
            },
            "total_stops": {
                "baseline": stats_base_stops,
                "v2i": stats_v2i_stops,
                "difference_mean": diff_mean_stops,
                "reduction_percent": pct_stops,
            },
            "preemption_delay": stats_v2i_delay,
        }

    def generate_text_report(self) -> str:
        """Generate a clean, professional human-readable markdown / text report."""
        summary = self.compute_summary()
        tt = summary["travel_time"]
        sw = summary["signal_wait_time"]
        cl = summary["clearance_time"]
        st = summary["total_stops"]
        pd = summary["preemption_delay"]

        lines = [
            "=================================================================================",
            "             PHASE 10 — V2I SMART INTERSECTION QUANTITATIVE BENEFIT REPORT      ",
            "=================================================================================",
            f"Total Controlled Trials: {summary['total_trials']} paired runs",
            f"Baseline Status: {summary['baseline_completed']}/{summary['total_trials']} Completed, {summary['baseline_timeouts']} Timeouts",
            f"V2I-Enabled Status: {summary['v2i_completed']}/{summary['total_trials']} Completed, {summary['v2i_timeouts']} Timeouts",
            "",
            "---------------------------------------------------------------------------------",
            "  METRIC                        BASELINE (NO V2I)      V2I ENABLED       CHANGE / BENEFIT",
            "---------------------------------------------------------------------------------",
            f"  Mean Travel Time               {tt['baseline']['mean']:6.2f} s           {tt['v2i']['mean']:6.2f} s       {'-' if tt['difference_mean'] >= 0 else '+'}{abs(tt['difference_mean']):5.2f} s ({abs(tt['reduction_percent']):.1f}% {'reduction' if tt['difference_mean'] >= 0 else 'increase'})",
            f"  Median Travel Time             {tt['baseline']['median']:6.2f} s           {tt['v2i']['median']:6.2f} s",
            f"  Min / Max Travel Time          {tt['baseline']['min']:4.2f} / {tt['baseline']['max']:4.2f} s     {tt['v2i']['min']:4.2f} / {tt['v2i']['max']:4.2f} s",
            "",
            f"  Mean Signal Waiting Time       {sw['baseline']['mean']:6.2f} s           {sw['v2i']['mean']:6.2f} s       {'-' if sw['difference_mean'] >= 0 else '+'}{abs(sw['difference_mean']):5.2f} s ({abs(sw['reduction_percent']):.1f}% {'reduction' if sw['difference_mean'] >= 0 else 'increase'})",
            f"  Median Signal Waiting Time     {sw['baseline']['median']:6.2f} s           {sw['v2i']['median']:6.2f} s",
            f"  Min / Max Signal Wait          {sw['baseline']['min']:4.2f} / {sw['baseline']['max']:4.2f} s     {sw['v2i']['min']:4.2f} / {sw['v2i']['max']:4.2f} s",
            "",
            f"  Mean Intersection Clearance    {cl['baseline']['mean']:6.2f} s           {cl['v2i']['mean']:6.2f} s",
            f"  Mean Total Full Stops          {st['baseline']['mean']:6.2f} stops        {st['v2i']['mean']:6.2f} stops    {'-' if st['difference_mean'] >= 0 else '+'}{abs(st['difference_mean']):5.2f} ({abs(st['reduction_percent']):.1f}% {'reduction' if st['difference_mean'] >= 0 else 'increase'})",
            f"  Mean V2I Preemption Delay            N/A             {pd['mean']:6.2f} s       (Request -> Priority Green)",
            "---------------------------------------------------------------------------------",
            "",
            "DETAILED PAIRED TRIAL COMPARISONS:",
            "Trial | Baseline TT | V2I TT | TT Diff | Baseline Wait | V2I Wait | Wait Diff | Stops (B/V)",
            "--------------------------------------------------------------------------------------",
        ]

        for i in range(self.num_trials):
            b = self.baseline_results[i]
            v = self.v2i_results[i]

            b_tt_str = f"{b.total_travel_time:5.2f}s" if b.total_travel_time is not None else "TIMEOUT"
            v_tt_str = f"{v.total_travel_time:5.2f}s" if v.total_travel_time is not None else "TIMEOUT"

            if b.total_travel_time is not None and v.total_travel_time is not None:
                d_tt = b.total_travel_time - v.total_travel_time
                if abs(d_tt) < 0.005:
                    diff_tt = "  0.00s"
                elif d_tt > 0:
                    diff_tt = f"-{d_tt:4.2f}s"
                else:
                    diff_tt = f"+{abs(d_tt):4.2f}s"
            else:
                diff_tt = "  N/A  "

            d_wait = b.signal_wait_time - v.signal_wait_time
            if abs(d_wait) < 0.005:
                diff_wait = "  0.00s"
            elif d_wait > 0:
                diff_wait = f"-{d_wait:4.2f}s"
            else:
                diff_wait = f"+{abs(d_wait):4.2f}s"

            lines.append(
                f" #{i+1:02d}  | {b_tt_str:11s} | {v_tt_str:6s} | {diff_tt:7s} | {b.signal_wait_time:10.2f}s  | {v.signal_wait_time:5.2f}s   | {diff_wait:9s} | {b.total_stops}/{v.total_stops}"
            )

        lines.append("=================================================================================")
        return "\n".join(lines)

    def export_json(self, filepath: str) -> None:
        """Export comprehensive trial results and statistical comparison to JSON."""
        data = {
            "summary": self.compute_summary(),
            "baseline_trials": [r.to_dict() for r in self.baseline_results],
            "v2i_trials": [r.to_dict() for r in self.v2i_results],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, filepath: str) -> None:
        """Export paired trial comparisons to a tabular CSV file."""
        fieldnames = [
            "trial_id", "seed",
            "baseline_status", "v2i_status",
            "baseline_travel_time", "v2i_travel_time", "travel_time_difference",
            "baseline_signal_wait", "v2i_signal_wait", "signal_wait_difference",
            "baseline_traffic_wait", "v2i_traffic_wait",
            "baseline_clearance_time", "v2i_clearance_time",
            "baseline_stops", "v2i_stops",
            "v2i_request_time", "v2i_reception_time", "v2i_preemption_time", "v2i_preemption_delay",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(self.num_trials):
                b = self.baseline_results[i]
                v = self.v2i_results[i]

                tt_diff = (
                    round(b.total_travel_time - v.total_travel_time, 3)
                    if (b.total_travel_time is not None and v.total_travel_time is not None)
                    else None
                )
                sw_diff = round(b.signal_wait_time - v.signal_wait_time, 3)

                writer.writerow({
                    "trial_id": b.trial_id,
                    "seed": b.seed,
                    "baseline_status": b.status,
                    "v2i_status": v.status,
                    "baseline_travel_time": b.total_travel_time,
                    "v2i_travel_time": v.total_travel_time,
                    "travel_time_difference": tt_diff,
                    "baseline_signal_wait": b.signal_wait_time,
                    "v2i_signal_wait": v.signal_wait_time,
                    "signal_wait_difference": sw_diff,
                    "baseline_traffic_wait": b.traffic_wait_time,
                    "v2i_traffic_wait": v.traffic_wait_time,
                    "baseline_clearance_time": b.intersection_clearance_time,
                    "v2i_clearance_time": v.intersection_clearance_time,
                    "baseline_stops": b.total_stops,
                    "v2i_stops": v.total_stops,
                    "v2i_request_time": v.emergency_request_time,
                    "v2i_reception_time": v.rsu_reception_time,
                    "v2i_preemption_time": v.preemption_activation_time,
                    "v2i_preemption_delay": v.preemption_delay,
                })
