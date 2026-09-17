"""Trace the real simulation state step-by-step for 13 seconds (Seed 42).
Prints exact timestamps, AI lifecycle transitions, C-01 telemetry samples,
predicted waypoints, and conflict detections.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import time
import pygame

from v2i.smart_signal import SmartTrafficSignal, CyclePhase
from v2i.traffic_manager import TrafficManager
from v2i.v2i_manager import V2IManager
from v2i.v2v_inter_manager import IntersectionV2VManager
from v2i.event_logger import EventLogger, EventCategory
import v2i.v2i_renderer as renderer

def main():
    random.seed(42)
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()

    signal_controller = SmartTrafficSignal()

    traffic_manager = TrafficManager(
        center_x=renderer.CX,
        center_y=renderer.CY,
        road_width=renderer.ROAD_WIDTH,
        stop_line_dist=renderer.STOP_LINE_DIST,
        ai_predictor=None,
    )
    traffic_manager.ai_loading = True

    import threading
    _ai_ready_event = threading.Event()
    _ai_holder = {"predictor": None, "error": None}

    def _async_load_ai():
        print("[DEBUG THREAD] AI loader thread started...")
        try:
            t0 = time.time()
            from ai.intersection_ai.predictor import IntersectionTrajectoryPredictor
            print(f"[DEBUG THREAD] Imported predictor in {time.time()-t0:.2f}s, constructing...")
            t1 = time.time()
            pred = IntersectionTrajectoryPredictor(base_dir=".")
            print(f"[DEBUG THREAD] Constructed predictor in {time.time()-t1:.2f}s, setting event...")
            _ai_holder["predictor"] = pred
            _ai_ready_event.set()
            print(f"[DEBUG THREAD] Ready event set! Total: {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"[DEBUG THREAD] Exception in AI loader: {e}")
            import traceback
            traceback.print_exc()
            _ai_holder["error"] = str(e)
            _ai_ready_event.set()

    ai_loader_thread = threading.Thread(target=_async_load_ai, name="AIPredictorLoader", daemon=True)
    ai_loader_thread.start()

    v2i_channel = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=0.0)
    v2v_channel = IntersectionV2VManager(v2v_range=200.0, latency=0.020, packet_loss_rate=0.0)
    event_logger = EventLogger(max_events=14)

    sim_time = 0.0
    FPS = 60
    DT = 1.0 / FPS

    last_ai_state = None
    last_c01_samples = -1

    for frame in range(13 * 60):
        # AI handoff check exactly as in v2i_main.py
        if _ai_ready_event.is_set():
            _ai_ready_event.clear()
            if _ai_holder["predictor"] is not None:
                traffic_manager.set_ai_predictor(_ai_holder["predictor"])
                print(f"[{sim_time:.2f}s] AI PREDICTOR ATTACHED to TrafficManager!")
                event_logger.add_event(
                    timestamp=sim_time,
                    category=EventCategory.SYSTEM,
                    event_type="AI",
                    message="AI PREDICTOR ONLINE (1.50s HORIZON)",
                )

        signal_controller.update(DT)
        sim_time += DT

        traffic_manager.update(
            dt=DT,
            signal_controller=signal_controller,
            sim_time=sim_time,
            v2i_channel=v2i_channel,
            rsu_pos=(renderer.CX, renderer.CY),
            v2v_inter_manager=v2v_channel,
        )

        event_logger.observe_system(
            sim_time=sim_time,
            signal_controller=signal_controller,
            traffic_manager=traffic_manager,
            v2i_channel=v2i_channel,
            v2v_inter_channel=v2v_channel,
        )

        # Inspect AI status
        pred = getattr(traffic_manager.ambulance, "latest_ai_prediction", None) if traffic_manager.ambulance else None
        if pred is None:
            pred = getattr(traffic_manager, "latest_ai_prediction", None)

        predictor = traffic_manager.ai_predictor
        c01_samples = len(predictor.history_buffers.get("C-01", [])) if (predictor and hasattr(predictor, "history_buffers")) else 0

        # Current AI state
        if predictor is None and traffic_manager.ai_loading:
            current_state = "INITIALIZING"
        elif predictor is None:
            current_state = "UNAVAILABLE"
        elif pred is None or not getattr(pred, "prediction_available", False):
            current_state = f"COLLECTING DATA ({c01_samples}/5)"
        elif getattr(pred, "predicted_conflict", False):
            current_state = "ACTIVE (PREDICTED_CONFLICT)"
        else:
            current_state = "ACTIVE (PREDICTED_SAFE)"

        if current_state != last_ai_state or (c01_samples != last_c01_samples and c01_samples <= 5):
            print(f"[{sim_time:5.2f}s] AI STATE: {current_state:30s} | C-01 pkts: {c01_samples}")
            if pred and getattr(pred, "prediction_available", False):
                wps = getattr(pred, "waypoints", [])
                print(f"         Waypoints ({len(wps)}): {[(round(w.x, 1), round(w.y, 1)) for w in wps]}")
                print(f"         Min Distance: {pred.predicted_min_distance:.1f}px, Risk: {pred.ai_risk_state}")
            last_ai_state = current_state
            last_c01_samples = c01_samples
        clock.tick(FPS)

    print(f"\n[DONE] Simulation finished at sim_time={sim_time:.2f}s")
    print("Waiting for ai_loader_thread to finish...")
    ai_loader_thread.join(timeout=5.0)
    print(f"ai_loader_thread is_alive: {ai_loader_thread.is_alive()}, _ai_holder: {_ai_holder}")

if __name__ == "__main__":
    main()
