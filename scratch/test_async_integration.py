"""Verification script for Pre-Phase 13E AI Asynchronous Runtime Race Fix."""
import os
import random
import sys
import threading
import time

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from ai.intersection_ai.predictor import IntersectionTrajectoryPredictor
from v2i.smart_signal import SmartTrafficSignal, CyclePhase
from v2i.traffic_manager import TrafficManager, SOUTH_PRIMARY_LANE_X, SOUTH_OVERTAKE_LANE_X
from v2i.v2i_manager import V2IManager
from v2i.v2v_inter_manager import IntersectionV2VManager
from v2i.event_logger import EventLogger, EventCategory

def run_simulation_trace():
    random.seed(42)

    signal_controller = SmartTrafficSignal("INT-01", 560, 415)
    traffic_manager = TrafficManager(center_x=560, center_y=415, road_width=130, stop_line_dist=75, ai_predictor=None)
    traffic_manager.ai_loading = True

    v2i_channel = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=0.0)
    v2v_channel = IntersectionV2VManager(v2v_range=200.0, latency=0.020, packet_loss_rate=0.0)
    event_logger = EventLogger(max_events=20)

    _ai_ready_event = threading.Event()
    _ai_holder = {"predictor": None, "error": None}

    def _async_load():
        try:
            p = IntersectionTrajectoryPredictor(base_dir=".")
            _ai_holder["predictor"] = p
            _ai_ready_event.set()
        except Exception as e:
            _ai_holder["error"] = str(e)
            _ai_ready_event.set()

    load_thread = threading.Thread(target=_async_load, daemon=True)
    load_thread.start()

    import pygame
    clock = pygame.time.Clock()
    FPS = 60
    dt = 1.0 / FPS
    sim_time = 0.0

    c01_spawn_time = None
    ai_attached_time = None
    collecting_timestamps = {}
    active_time = None
    conflict_time = None
    ttc_critical_time = None
    maneuver_time = None

    print(f"Starting simulation trace (dt={dt:.4f}s)...")
    for step in range(1200): # 20 simulated seconds
        # Check thread-safe handoff
        if _ai_ready_event.is_set():
            _ai_ready_event.clear()
            if _ai_holder["predictor"] is not None:
                traffic_manager.set_ai_predictor(_ai_holder["predictor"])
                ai_attached_time = sim_time
                print(f"[t={sim_time:.2f}s] AI PREDICTOR ATTACHED to TrafficManager")
            elif _ai_holder["error"] is not None:
                traffic_manager.ai_loading = False
                traffic_manager.ai_error = _ai_holder["error"]
                print(f"[t={sim_time:.2f}s] AI LOAD ERROR: {_ai_holder['error']}")

        signal_controller.update(dt)
        sim_time += dt

        traffic_manager.update(
            dt=dt,
            signal_controller=signal_controller,
            sim_time=sim_time,
            v2i_channel=v2i_channel,
            rsu_pos=(730, 285),
            v2v_inter_manager=v2v_channel,
        )

        # Deliver in-flight V2I packets to RSU
        delivered_msgs = v2i_channel.deliver_to_rsu(sim_time)
        for msg in delivered_msgs:
            signal_controller.receive_emergency_request(msg, current_time=sim_time, auto_preempt=True)

        signal_controller.update_traffic_analysis(
            civilian_vehicles=traffic_manager.vehicles,
            ambulance=traffic_manager.ambulance,
            current_time=sim_time,
        )

        # Track C-01
        c01 = next((v for v in traffic_manager.vehicles if v.vehicle_id == "C-01"), None)
        if c01 and c01_spawn_time is None:
            c01_spawn_time = sim_time
            print(f"[t={sim_time:.2f}s] C-01 SPAWNED at y={c01.y:.1f}")

        # Track history count
        if traffic_manager.ai_predictor:
            buf = traffic_manager.ai_predictor.history_buffers.get("C-01", [])
            cnt = len(buf)
            if cnt > 0 and cnt not in collecting_timestamps:
                collecting_timestamps[cnt] = sim_time
                print(f"[t={sim_time:.2f}s] AI COLLECTING DATA ({cnt}/5)")

        # Track active prediction
        amb = traffic_manager.ambulance
        pred = (amb.latest_ai_prediction if amb and amb.latest_ai_prediction else traffic_manager.latest_ai_prediction)
        if pred and pred.prediction_available:
            if active_time is None:
                active_time = sim_time
                print(f"[t={sim_time:.2f}s] AI ACTIVE! Sender: {pred.sender_id}, Waypoints: {len(pred.waypoints)}, Risk: {pred.ai_risk_state}, Latency: {pred.inference_time_ms:.2f}ms")
                for i, wp in enumerate(pred.waypoints):
                    print(f"   wp[{i}]: t=+{wp.horizon_offset_s:.2f}s, x={wp.x:.1f}, y={wp.y:.1f}")

            if pred.predicted_conflict and conflict_time is None:
                conflict_time = sim_time
                print(f"[t={sim_time:.2f}s] AI PREDICTED CONFLICT! Min dist: {pred.predicted_min_distance:.1f}px")

        # Track ambulance
        amb = traffic_manager.ambulance
        if amb:
            if amb.v2v_risk_state == "CRITICAL" and ttc_critical_time is None:
                ttc_critical_time = sim_time
                print(f"[t={sim_time:.2f}s] TTC CRITICAL! TTC: {amb.v2v_ttc:.2f}s")

            if amb.is_v2v_evading and maneuver_time is None:
                maneuver_time = sim_time
                print(f"[t={sim_time:.2f}s] AMBULANCE EVASIVE MANEUVER to x={amb.target_lane_x:.1f}")

        clock.tick(FPS)

    print("\n--- SUMMARY REPORT ---")
    print(f"C-01 spawn time: {c01_spawn_time:.2f}s" if c01_spawn_time else "C-01 not spawned")
    print(f"AI attached time: {ai_attached_time:.2f}s" if ai_attached_time else "AI not attached")
    print(f"AI ACTIVE time: {active_time:.2f}s" if active_time else "AI never active")
    print(f"AI Conflict time: {conflict_time:.2f}s" if conflict_time else "AI no conflict")
    print(f"TTC Critical time: {ttc_critical_time:.2f}s" if ttc_critical_time else "TTC no critical")
    if conflict_time and ttc_critical_time:
        lead = ttc_critical_time - conflict_time
        print(f"AI Lead Time: {lead:+.2f}s (AI warning before TTC critical)")

if __name__ == "__main__":
    run_simulation_trace()
