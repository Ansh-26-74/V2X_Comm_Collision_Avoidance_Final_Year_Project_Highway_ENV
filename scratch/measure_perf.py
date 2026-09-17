"""Performance verification script for Pre-Phase 13E fix.
Measures:
  1. Startup time to first visible frame
  2. AI background initialization time
  3. Steady-state FPS
  4. AI inference latency
"""

import time
import os
import sys
sys.path.insert(0, os.path.abspath("."))

t_start = time.perf_counter()
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

# Initialize headless pygame
pygame.init()
pygame.font.init()
screen = pygame.Surface((1280, 720))
clock = pygame.time.Clock()

import v2i.v2i_renderer as renderer
from v2i.smart_signal import SmartTrafficSignal
from v2i.traffic_manager import TrafficManager
from v2i.v2i_manager import V2IManager
from v2i.event_logger import EventLogger
import threading

signal = SmartTrafficSignal()
v2i = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=0.0)
logger = EventLogger(max_events=20)

t_tm_start = time.perf_counter()
tm = TrafficManager(
    center_x=renderer.CX,
    center_y=renderer.CY,
    road_width=renderer.ROAD_WIDTH,
    stop_line_dist=renderer.STOP_LINE_DIST,
)
t_tm_end = time.perf_counter()

font_small = pygame.font.SysFont("consolas", 12, bold=True)
font_normal = pygame.font.SysFont("consolas", 14)
font_bold = pygame.font.SysFont("consolas", 16, bold=True)
font_title = pygame.font.SysFont("consolas", 18, bold=True)

# Frame 0 render (first visible frame)
renderer.draw_four_way_intersection(screen)
renderer.draw_conflict_zone(screen, None, font_small)
renderer.draw_v2i_communication(screen, v2i, tm.ambulance, font_small, signal_controller=signal)
renderer.draw_civilian_vehicles(screen, tm.vehicles, font_small, font_normal)
renderer.draw_all_signals(screen, signal, font_normal, font_bold)
renderer.draw_rsu_station(screen, font_small)

renderer.draw_hud(
    screen=screen,
    signal_controller=signal,
    traffic_manager=tm,
    font_title=font_title,
    font_bold=font_bold,
    font_normal=font_normal,
    font_small=font_small,
    sim_time=0.0,
    paused=False,
    v2i_channel=v2i,
    event_logger=logger,
    show_ai=True,
)

t_first_frame = time.perf_counter()

# Background AI initialization
t_ai_init_start = time.perf_counter()
from ai.intersection_ai.predictor import IntersectionTrajectoryPredictor
predictor = IntersectionTrajectoryPredictor()
tm.ai_predictor = predictor
t_ai_init_end = time.perf_counter()

# AI inference latency
for i in range(5):
    predictor.add_telemetry({
        "sender_id": "C-01",
        "timestamp": 0.1 * (i + 1),
        "x": 592.0,
        "y": 500.0 - 20.0 * i,
        "vx": 0.0,
        "vy": -20.0,
        "speed": 20.0,
        "heading": -1.57,
        "hazard_status": "NORMAL",
    })

t_infer_start = time.perf_counter()
pred_result = predictor.predict(
    sender_id="C-01",
    ambulance_pos=(592.0, 600.0),
    ambulance_speed=25.0,
)
t_infer_end = time.perf_counter()

# Steady-state FPS simulation (120 frames)
fps_samples = []
for frame in range(120):
    t0 = time.perf_counter()
    tm.update(1.0 / 60.0, signal)
    signal.update(1.0 / 60.0)
    renderer.draw_four_way_intersection(screen)
    renderer.draw_civilian_vehicles(screen, tm.vehicles, font_small, font_normal)
    renderer.draw_all_signals(screen, signal, font_normal, font_bold)
    renderer.draw_hud(
        screen=screen,
        signal_controller=signal,
        traffic_manager=tm,
        font_title=font_title,
        font_bold=font_bold,
        font_normal=font_normal,
        font_small=font_small,
        sim_time=frame * (1.0 / 60.0),
        v2i_channel=v2i,
        event_logger=logger,
        show_ai=True,
    )
    t1 = time.perf_counter()
    dt_frame = t1 - t0
    if dt_frame > 0:
        fps_samples.append(1.0 / dt_frame)

mean_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0

print(f"RESULTS:")
print(f"Time to first visible frame: {t_first_frame - t_start:.2f}s")
print(f"TrafficManager creation time: {t_tm_end - t_tm_start:.2f}s")
print(f"AI initialization time: {t_ai_init_end - t_ai_init_start:.2f}s")
print(f"AI inference latency: {(t_infer_end - t_infer_start)*1000.0:.2f}ms")
print(f"Steady-state rendering FPS: {mean_fps:.1f} FPS (simulation overhead: {1000.0/mean_fps:.2f}ms/frame)")
