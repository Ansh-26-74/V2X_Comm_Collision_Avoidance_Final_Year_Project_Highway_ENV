"""Phase 10 Test Suite: Quantitative V2I Benefit Measurement & Benchmark System.

Verifies:
1. Metric Creation: MetricsCollector and TrialResult initialize correctly.
2. Measurement Start: The ambulance spawn time is captured exactly once as the measurement start.
3. Clearance Completion: Authoritative Phase 7 full-body clearance event ends the measurement.
4. Signal Waiting Time: Accrued strictly during WAITING_AT_RED transitions, excluding obstacle waits.
5. Stop Counting Deduplication: Consecutive frames with zero velocity record exactly 1 stop event.
6. V2I Timestamps: Emergency request, RSU reception, and preemption activation times are recorded.
7. Baseline Isolation: BASELINE_NO_V2I executes with emergency preemption completely disabled.
8. V2I Isolation: V2I_ENABLED activates the genuine Phase 4-8 emergency preemption mechanism.
9. Deterministic Repeatability: Identical seed and configuration produce bit-for-bit identical results.
10. Timeout Handling: Incomplete runs return status TIMEOUT without fabricated travel times.
11. Statistical Calculations: Mean, median, paired differences, and percentage reductions calculate accurately.
12. Export Integrity: JSON and CSV exports produce valid files with all required columns.
"""

import os
import json
import csv
import unittest

from v2i.metrics import (
    ExperimentMode,
    TrialStatus,
    TrialConfig,
    TrialResult,
    MetricsCollector,
    SimulationRunner,
    ExperimentSuite,
    ComparisonReport,
)
from v2i.smart_signal import SmartTrafficSignal, CyclePhase
from v2i.traffic_manager import TrafficManager, AmbulanceVehicle, AmbulanceState
from v2i.v2i_manager import V2IManager
import v2i.v2i_renderer as renderer


class TestPhase10Metrics(unittest.TestCase):
    """Unit and integration tests for Phase 10 metrics and benchmarking."""

    def setUp(self):
        self.config = TrialConfig(
            trial_id=1,
            seed=1001,
            ambulance_spawn_time=8.5,
            timeout=60.0,
            description="Test Config",
        )

    def test_01_metric_creation(self):
        """MetricsCollector and TrialConfig initialize with valid default state."""
        collector = MetricsCollector(mode=ExperimentMode.V2I_ENABLED, trial_config=self.config)
        self.assertEqual(collector.mode, ExperimentMode.V2I_ENABLED)
        self.assertEqual(collector.config.trial_id, 1)
        self.assertIsNone(collector.spawn_time)
        self.assertIsNone(collector.clearance_time)
        self.assertEqual(collector.signal_wait_time, 0.0)
        self.assertEqual(collector.traffic_wait_time, 0.0)
        self.assertEqual(collector.total_stops, 0)
        self.assertEqual(collector.signal_stops, 0)
        self.assertEqual(collector.traffic_stops, 0)

    def test_02_measurement_start(self):
        """Measurement start is captured exactly once upon ambulance spawn."""
        collector = MetricsCollector(mode=ExperimentMode.BASELINE_NO_V2I, trial_config=self.config)
        signal = SmartTrafficSignal()
        tm = TrafficManager(
            center_x=renderer.CX,
            center_y=renderer.CY,
            road_width=renderer.ROAD_WIDTH,
            stop_line_dist=renderer.STOP_LINE_DIST,
            ambulance_spawn_time=5.0,
        )

        # Step prior to spawn
        collector.step(sim_time=4.0, dt=1.0/60.0, signal_controller=signal, traffic_manager=tm)
        self.assertIsNone(collector.spawn_time)

        # Step at spawn time
        tm._spawn_ambulance()
        collector.step(sim_time=5.0, dt=1.0/60.0, signal_controller=signal, traffic_manager=tm)
        self.assertEqual(collector.spawn_time, 5.0)

        # Subsequent steps do NOT overwrite spawn_time
        collector.step(sim_time=6.0, dt=1.0/60.0, signal_controller=signal, traffic_manager=tm)
        self.assertEqual(collector.spawn_time, 5.0)

    def test_03_clearance_completion(self):
        """Clearance completion timestamp is captured and ends measurement."""
        collector = MetricsCollector(mode=ExperimentMode.BASELINE_NO_V2I, trial_config=self.config)
        signal = SmartTrafficSignal()
        tm = TrafficManager(
            center_x=renderer.CX,
            center_y=renderer.CY,
            road_width=renderer.ROAD_WIDTH,
            stop_line_dist=renderer.STOP_LINE_DIST,
            ambulance_spawn_time=2.0,
        )
        tm._spawn_ambulance()
        collector.step(sim_time=2.0, dt=0.1, signal_controller=signal, traffic_manager=tm)

        # Mark ambulance as cleared at t=10.0
        tm.ambulance.cleared = True
        collector.step(sim_time=10.0, dt=0.1, signal_controller=signal, traffic_manager=tm)
        self.assertEqual(collector.clearance_time, 10.0)

        # Finalized result reflects completion
        res = collector.finalize(sim_time=12.0)
        self.assertEqual(res.status, TrialStatus.COMPLETED.value)
        self.assertEqual(res.measurement_start, 2.0)
        self.assertEqual(res.measurement_end, 10.0)
        self.assertAlmostEqual(res.total_travel_time, 8.0, places=3)

    def test_04_signal_waiting_time(self):
        """Signal waiting time accumulates strictly during WAITING_AT_RED state."""
        collector = MetricsCollector(mode=ExperimentMode.BASELINE_NO_V2I, trial_config=self.config)
        signal = SmartTrafficSignal()
        tm = TrafficManager(
            center_x=renderer.CX,
            center_y=renderer.CY,
            road_width=renderer.ROAD_WIDTH,
            stop_line_dist=renderer.STOP_LINE_DIST,
        )
        tm._spawn_ambulance()
        amb = tm.ambulance

        # Stopped at RED for 10 frames of 0.1s
        amb.speed = 0.0
        amb.ambulance_state = AmbulanceState.WAITING_AT_RED
        for _ in range(10):
            collector.step(sim_time=1.0, dt=0.1, signal_controller=signal, traffic_manager=tm)

        self.assertAlmostEqual(collector.signal_wait_time, 1.0, places=3)
        self.assertEqual(collector.traffic_wait_time, 0.0)

        # Stopped behind obstacle (traffic wait) for 5 frames of 0.1s
        amb.ambulance_state = AmbulanceState.STOPPED
        for _ in range(5):
            collector.step(sim_time=2.0, dt=0.1, signal_controller=signal, traffic_manager=tm)

        self.assertAlmostEqual(collector.signal_wait_time, 1.0, places=3)
        self.assertAlmostEqual(collector.traffic_wait_time, 0.5, places=3)

    def test_05_stop_counting_deduplication(self):
        """Continuous stopped frames do not create multiple fake stop events."""
        collector = MetricsCollector(mode=ExperimentMode.BASELINE_NO_V2I, trial_config=self.config)
        signal = SmartTrafficSignal()
        tm = TrafficManager(
            center_x=renderer.CX,
            center_y=renderer.CY,
            road_width=renderer.ROAD_WIDTH,
            stop_line_dist=renderer.STOP_LINE_DIST,
        )
        tm._spawn_ambulance()
        amb = tm.ambulance

        # Initially moving
        amb.speed = 90.0
        collector.step(sim_time=1.0, dt=0.016, signal_controller=signal, traffic_manager=tm)
        self.assertEqual(collector.total_stops, 0)

        # Comes to a complete stop at RED for 60 consecutive frames
        amb.speed = 0.0
        amb.ambulance_state = AmbulanceState.WAITING_AT_RED
        for i in range(60):
            collector.step(sim_time=1.0 + i*0.016, dt=0.016, signal_controller=signal, traffic_manager=tm)

        self.assertEqual(collector.total_stops, 1, "Expected 1 deduplicated stop event")
        self.assertEqual(collector.signal_stops, 1)

        # Starts moving again
        amb.speed = 40.0
        collector.step(sim_time=2.0, dt=0.016, signal_controller=signal, traffic_manager=tm)

        # Comes to second stop
        amb.speed = 0.0
        amb.ambulance_state = AmbulanceState.STOPPED
        for i in range(30):
            collector.step(sim_time=3.0 + i*0.016, dt=0.016, signal_controller=signal, traffic_manager=tm)

        self.assertEqual(collector.total_stops, 2)
        self.assertEqual(collector.traffic_stops, 1)

    def test_06_v2i_timestamps(self):
        """V2I request, reception, and preemption activation timestamps are captured accurately."""
        collector = MetricsCollector(mode=ExperimentMode.V2I_ENABLED, trial_config=self.config)
        signal = SmartTrafficSignal()
        tm = TrafficManager(
            center_x=renderer.CX,
            center_y=renderer.CY,
            road_width=renderer.ROAD_WIDTH,
            stop_line_dist=renderer.STOP_LINE_DIST,
        )
        tm._spawn_ambulance()
        v2i = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=0.0)

        # Simulate transmission at t=3.0
        v2i.total_transmitted = 1
        collector.step(sim_time=3.0, dt=0.1, signal_controller=signal, traffic_manager=tm, v2i_channel=v2i)
        self.assertEqual(collector.emergency_request_time, 3.0)

        # Simulate delivery at t=3.2
        v2i.total_delivered = 1
        collector.step(sim_time=3.2, dt=0.1, signal_controller=signal, traffic_manager=tm, v2i_channel=v2i)
        self.assertEqual(collector.rsu_reception_time, 3.2)

        # Simulate preemption activation at t=3.3
        signal.preemption_requested = True
        collector.step(sim_time=3.3, dt=0.1, signal_controller=signal, traffic_manager=tm, v2i_channel=v2i)
        self.assertEqual(collector.preemption_activation_time, 3.3)

        res = collector.finalize(sim_time=4.0)
        self.assertAlmostEqual(res.preemption_delay, 0.3, places=3)

    def test_07_baseline_isolation(self):
        """BASELINE_NO_V2I runs with emergency preemption completely disabled."""
        cfg = TrialConfig(trial_id=1, seed=1001, ambulance_spawn_time=8.5, timeout=40.0)
        result = SimulationRunner.run_trial(ExperimentMode.BASELINE_NO_V2I, cfg)

        self.assertEqual(result.mode, ExperimentMode.BASELINE_NO_V2I.value)
        self.assertEqual(result.status, TrialStatus.COMPLETED.value)
        self.assertIsNone(result.emergency_request_time)
        self.assertIsNone(result.rsu_reception_time)
        self.assertIsNone(result.preemption_activation_time)
        self.assertIsNone(result.preemption_delay)
        # Baseline must wait at red light when arriving during EW green
        self.assertGreater(result.signal_wait_time, 5.0)

    def test_08_v2i_isolation(self):
        """V2I_ENABLED runs with emergency preemption active, capturing V2I timestamps."""
        cfg = TrialConfig(trial_id=1, seed=1001, ambulance_spawn_time=8.5, timeout=40.0)
        result = SimulationRunner.run_trial(ExperimentMode.V2I_ENABLED, cfg)

        self.assertEqual(result.mode, ExperimentMode.V2I_ENABLED.value)
        self.assertEqual(result.status, TrialStatus.COMPLETED.value)
        self.assertIsNotNone(result.emergency_request_time)
        self.assertIsNotNone(result.rsu_reception_time)
        self.assertIsNotNone(result.preemption_activation_time)
        self.assertIsNotNone(result.preemption_delay)
        self.assertGreater(result.preemption_delay, 0.0)

    def test_09_deterministic_repeatability(self):
        """Identical trial configuration and seed produce bit-for-bit identical results."""
        cfg = TrialConfig(trial_id=1, seed=42, ambulance_spawn_time=8.5, timeout=30.0)
        res1 = SimulationRunner.run_trial(ExperimentMode.V2I_ENABLED, cfg)
        res2 = SimulationRunner.run_trial(ExperimentMode.V2I_ENABLED, cfg)

        self.assertEqual(res1.total_travel_time, res2.total_travel_time)
        self.assertEqual(res1.signal_wait_time, res2.signal_wait_time)
        self.assertEqual(res1.intersection_clearance_time, res2.intersection_clearance_time)
        self.assertEqual(res1.total_stops, res2.total_stops)

    def test_10_timeout_handling(self):
        """Incomplete trial returns status TIMEOUT without fabricating travel time."""
        # Set artificially small timeout (2.0s) so ambulance cannot reach intersection
        cfg = TrialConfig(trial_id=99, seed=123, ambulance_spawn_time=1.0, timeout=2.0)
        res = SimulationRunner.run_trial(ExperimentMode.BASELINE_NO_V2I, cfg)

        self.assertEqual(res.status, TrialStatus.TIMEOUT.value)
        self.assertIsNone(res.total_travel_time)
        self.assertIsNone(res.intersection_clearance_time)

    def test_11_statistical_calculations(self):
        """ComparisonReport correctly computes mean, median, differences, and percentage reductions."""
        # Mock results
        b1 = TrialResult("BASELINE_NO_V2I", 1, 101, "COMPLETED", 0.0, 20.0, 20.0, 10.0, 0.0, 2.0, 1, 1, 0)
        b2 = TrialResult("BASELINE_NO_V2I", 2, 102, "COMPLETED", 0.0, 10.0, 10.0, 4.0, 0.0, 2.0, 1, 1, 0)
        v1 = TrialResult("V2I_ENABLED", 1, 101, "COMPLETED", 0.0, 12.0, 12.0, 2.0, 0.0, 2.0, 1, 1, 0)
        v2 = TrialResult("V2I_ENABLED", 2, 102, "COMPLETED", 0.0, 8.0, 8.0, 2.0, 0.0, 2.0, 0, 0, 0)

        report = ComparisonReport([b1, b2], [v1, v2])
        summary = report.compute_summary()

        # Mean travel time: baseline = 15.0, v2i = 10.0 -> diff = 5.0, reduction = (5/15)*100 = 33.33%
        self.assertAlmostEqual(summary["travel_time"]["baseline"]["mean"], 15.0)
        self.assertAlmostEqual(summary["travel_time"]["v2i"]["mean"], 10.0)
        self.assertAlmostEqual(summary["travel_time"]["difference_mean"], 5.0)
        self.assertAlmostEqual(summary["travel_time"]["reduction_percent"], 33.333, places=2)

        # Mean signal wait: baseline = 7.0, v2i = 2.0 -> diff = 5.0, reduction = (5/7)*100 = 71.43%
        self.assertAlmostEqual(summary["signal_wait_time"]["baseline"]["mean"], 7.0)
        self.assertAlmostEqual(summary["signal_wait_time"]["v2i"]["mean"], 2.0)
        self.assertAlmostEqual(summary["signal_wait_time"]["difference_mean"], 5.0)
        self.assertAlmostEqual(summary["signal_wait_time"]["reduction_percent"], 71.428, places=2)

    def test_12_export_integrity(self):
        """JSON and CSV exports produce valid files with all required columns."""
        b = TrialResult("BASELINE_NO_V2I", 1, 101, "COMPLETED", 0.0, 20.0, 20.0, 10.0, 0.0, 2.0, 1, 1, 0)
        v = TrialResult("V2I_ENABLED", 1, 101, "COMPLETED", 0.0, 12.0, 12.0, 2.0, 0.0, 2.0, 1, 1, 0,
                        emergency_request_time=4.0, rsu_reception_time=4.2, preemption_activation_time=4.3, preemption_delay=0.3)
        report = ComparisonReport([b], [v])

        json_path = "test_export.json"
        csv_path = "test_export.csv"
        try:
            report.export_json(json_path)
            report.export_csv(csv_path)

            self.assertTrue(os.path.exists(json_path))
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertIn("summary", data)
                self.assertIn("baseline_trials", data)
                self.assertIn("v2i_trials", data)

            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 1)
                self.assertIn("travel_time_difference", rows[0])
                self.assertIn("signal_wait_difference", rows[0])
                self.assertIn("v2i_preemption_delay", rows[0])
        finally:
            if os.path.exists(json_path):
                os.remove(json_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()
