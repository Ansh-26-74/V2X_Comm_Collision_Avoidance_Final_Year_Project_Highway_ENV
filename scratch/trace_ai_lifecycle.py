import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import threading

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

import v2i_main
import v2i.v2i_renderer as renderer
from v2i.smart_signal import SmartTrafficSignal
from v2i.traffic_manager import TrafficManager
from v2i.v2i_manager import V2IManager
from v2i.v2v_inter_manager import IntersectionV2VManager
from v2i.event_logger import EventLogger

pygame.init()
pygame.font.init()
screen = pygame.Surface((1280, 720))

traffic_manager = TrafficManager(
    center_x=renderer.CX,
    center_y=renderer.CY,
    road_width=renderer.ROAD_WIDTH,
    stop_line_dist=renderer.STOP_LINE_DIST,
    ai_predictor=None,
)
traffic_manager.ai_loading = True

loader_error = None
loader_done = False

def _async_load_ai() -> None:
    global loader_error, loader_done
    t0 = time.perf_counter()
    try:
        print("[TEST_THREAD] Starting IntersectionTrajectoryPredictor...")
        from ai.intersection_ai.predictor import IntersectionTrajectoryPredictor
        pred = IntersectionTrajectoryPredictor(base_dir=".")
        traffic_manager.ai_predictor = pred
        traffic_manager.ai_loading = False
        loader_done = True
        print(f"[TEST_THREAD] AI Trajectory Predictor online in {time.perf_counter()-t0:.2f}s!")
    except Exception as e:
        loader_error = e
        traffic_manager.ai_loading = False
        print(f"[TEST_THREAD] AI load error: {e}")

th = threading.Thread(target=_async_load_ai, name="TestAILoader", daemon=True)
th.start()

signal = SmartTrafficSignal()
v2i = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=0.0)
v2v = IntersectionV2VManager(v2v_range=200.0, latency=0.020, packet_loss_rate=0.0)
logger = EventLogger(max_events=14)

fps = 60
dt = 1.0 / fps
sim_time = 0.0

print("\n--- Running Simulation Loop for 10 seconds (600 frames) ---")
for frame in range(600):
    sim_time += dt
    traffic_manager.update(
        dt=dt,
        signal_controller=signal,
        sim_time=sim_time,
        v2i_channel=v2i,
        rsu_pos=renderer.RSU_POS,
        v2v_inter_manager=v2v,
    )
    signal.update(dt)

    amb = traffic_manager.ambulance
    ai_pred = getattr(amb, "latest_ai_prediction", None) if amb else None
    pred_avail = getattr(ai_pred, "prediction_available", False) if ai_pred else False
    num_samples = len(traffic_manager.ai_predictor.history_buffers.get("C-01", [])) if traffic_manager.ai_predictor else 0

    if frame % 60 == 0 or pred_avail:
        print(f"t={sim_time:04.1f}s | ai_loading={traffic_manager.ai_loading} | predictor={traffic_manager.ai_predictor is not None} | C-01 samples={num_samples} | pred_avail={pred_avail} | risk={getattr(ai_pred, 'ai_risk_state', 'NONE')}")

    time.sleep(0.016)

th.join(timeout=1.0)
print(f"Final: loader_done={loader_done}, loader_error={loader_error}")
