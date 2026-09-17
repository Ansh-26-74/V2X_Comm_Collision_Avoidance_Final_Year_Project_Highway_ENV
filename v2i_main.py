"""Phase 1: V2I Smart Intersection Demonstration — 4-Way PLUS Intersection & Signals.

Demonstrates:
  - Realistic 4-Way PLUS (+) urban road intersection geometry.
  - Four coordinated traffic signals (North, South, East, West).
  - Coordinated normal cycle state machine:
      N/S GREEN (8s) -> N/S YELLOW (3s) -> E/W GREEN (8s) -> E/W YELLOW (3s) -> Repeat
  - Absolute enforcement of safety mutex: conflicting movements are NEVER simultaneously GREEN.
  - Visible RSU / Smart Infrastructure with active V2X status beacon.
  - Interactive controls: [SPACE] pause/resume, [R] reset cycle, [ESC/Q] exit.

Run with:
    python v2i_main.py

This file does NOT import from or modify any V2V module.
The existing V2V scenario (main.py) remains completely unaffected.
"""

import logging
import sys
import pygame

from v2i.smart_signal import SmartTrafficSignal, CyclePhase
from v2i.traffic_manager import TrafficManager
from v2i.v2i_manager import V2IManager
from v2i.event_logger import EventLogger
import v2i.v2i_renderer as renderer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Scene & Window Configuration
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_WIDTH  = 1200
WINDOW_HEIGHT = 750
FPS           = 60
SIM_FREQ      = 60
DT            = 1.0 / SIM_FREQ


def main() -> None:
    logger.info("=================================================================")
    logger.info("  V2I SMART INTERSECTION — PHASE 8: EMERGENCY PRIORITY TERMINATION")
    logger.info("=================================================================")
    logger.info("4-Way Urban Intersection + Autonomous Civilian Traffic + Controlled Recovery")

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("V2I Smart Intersection — Emergency Priority Termination & Recovery (Phase 8)")
    clock = pygame.time.Clock()

    # Fonts
    font_title  = pygame.font.SysFont("consolas", 20, bold=True)
    font_bold   = pygame.font.SysFont("consolas", 16, bold=True)
    font_normal = pygame.font.SysFont("consolas", 15)
    font_small  = pygame.font.SysFont("consolas", 13, bold=True)
    font_timer  = pygame.font.SysFont("consolas", 14, bold=True)

    # Instantiate RSU / Smart Traffic Signal Controller
    signal_controller = SmartTrafficSignal(
        intersection_id="INT-01",
        intersection_x=renderer.CX,
        intersection_y=renderer.CY,
        phase_timings={
            CyclePhase.NS_GREEN_EW_RED: 8.0,
            CyclePhase.NS_YELLOW_EW_RED: 3.0,
            CyclePhase.NS_RED_EW_GREEN: 8.0,
            CyclePhase.NS_RED_EW_YELLOW: 3.0,
        },
    )

    # Instantiate Civilian Traffic Manager (Phase 2 & 3)
    traffic_manager = TrafficManager(
        center_x=renderer.CX,
        center_y=renderer.CY,
        road_width=renderer.ROAD_WIDTH,
        stop_line_dist=renderer.STOP_LINE_DIST,
    )

    # Realistic V2I communication channel (Phase 4: range 400px, 200ms latency)
    v2i_channel = V2IManager(v2i_range=400.0, latency=0.2, packet_loss_rate=0.0)

    # Phase 9: Event Logger for deterministic chronological audit trail
    event_logger = EventLogger(max_events=14)

    sim_time = 0.0
    paused = False
    running = True

    logger.info("[INIT] Smart 4-Way Intersection initialized.")
    logger.info("[INIT] Signal Controller online. Starting normal coordinated cycle.")
    logger.info(f"[INIT] V2I DSRC Channel active (Range: {int(v2i_channel.v2i_range)}px, Latency: {int(v2i_channel.latency*1000)}ms).")
    logger.info("[INIT] Phase 9 Observability, Event Timeline & Telemetry active.")

    last_phase = signal_controller.current_phase

    while running:
        dt = clock.tick(FPS) / 1000.0

        # ── Event Handling ───────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    state_str = "PAUSED" if paused else "RESUMED"
                    logger.info(f"[USER] Simulation {state_str}")
                elif event.key == pygame.K_r:
                    signal_controller.reset()
                    traffic_manager.reset()
                    v2i_channel.reset()
                    event_logger.clear()
                    sim_time = 0.0
                    logger.info("[USER] Signal Controller, Traffic, V2I Channel, and Event Log reset to initial state")

        # ── Simulation Step ──────────────────────────────────────────────────
        if not paused:
            # Advance signal state machine (unmodified normal cycle)
            signal_controller.update(dt)
            sim_time += dt

            # Advance traffic manager & vehicles (AMB-01 handles V2I periodic transmission)
            traffic_manager.update(
                dt=dt,
                signal_controller=signal_controller,
                sim_time=sim_time,
                v2i_channel=v2i_channel,
                rsu_pos=renderer.RSU_POS,
            )

            # Phase 8: Authoritative emergency clearance event consumption
            if traffic_manager.ambulance_cleared_events and signal_controller.current_phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
                signal_controller.notify_ambulance_cleared("AMB-01")

            # Deliver in-flight packets that have reached their latency delivery time
            delivered_msgs = v2i_channel.deliver_to_rsu(sim_time)
            for msg in delivered_msgs:
                accepted = signal_controller.receive_emergency_request(msg, current_time=sim_time, auto_preempt=True)
                if accepted:
                    logger.info(
                        f"[V2I COMM] RSU-01 accepted {msg.get('type')} from {msg.get('vehicleId')} | "
                        f"Approach: {msg.get('approach')}, ETA: {msg.get('eta')}s, Speed: {msg.get('speed')}px/s | "
                        f"Action: SAFE EMERGENCY PREEMPTION INITIATED (Phase 6)"
                    )

            # Continuous RSU traffic & conflict analysis (Phase 5)
            signal_controller.update_traffic_analysis(
                civilian_vehicles=traffic_manager.vehicles,
                ambulance=traffic_manager.ambulance,
                current_time=sim_time,
            )

            # Update channel packet flight animations
            v2i_channel.update_animations(sim_time)

            # Phase 9: Observe real system state changes & log deterministic events
            event_logger.observe_system(
                sim_time=sim_time,
                signal_controller=signal_controller,
                traffic_manager=traffic_manager,
                v2i_channel=v2i_channel,
            )

            # Log phase transitions
            if signal_controller.current_phase != last_phase:
                last_phase = signal_controller.current_phase
                logger.info(
                    f"[RSU] New Phase: {last_phase.value} | "
                    f"Signals: N={signal_controller.get_signal('NORTH').value}, "
                    f"S={signal_controller.get_signal('SOUTH').value}, "
                    f"E={signal_controller.get_signal('EAST').value}, "
                    f"W={signal_controller.get_signal('WEST').value}"
                )

            # Advance renderer animations (beacon pulse, etc.)
            renderer.tick_renderer(dt)

        # ── Rendering Pipeline ───────────────────────────────────────────────
        # 1. 4-Way PLUS Intersection road geometry, asphalt, sidewalks, lane markings, stop lines
        renderer.draw_four_way_intersection(screen)

        # 2. Phase 5 Central Intersection Conflict Zone overlay (highlighting real-time occupancy)
        renderer.draw_conflict_zone(screen, signal_controller.latest_analysis, font_small)

        # 3. V2I wireless communication range zone, connection link, flying packets, & infrastructure preemption line
        renderer.draw_v2i_communication(screen, v2i_channel, traffic_manager.ambulance, font_small, signal_controller=signal_controller)

        # 4. Civilian vehicles & Phase 5 conflict status halos
        renderer.draw_civilian_vehicles(screen, traffic_manager.vehicles, font_small, font_timer)
        renderer.draw_vehicle_conflict_highlights(screen, traffic_manager.vehicles, font_small)

        # 5. Emergency Vehicle AMB-01 (Phase 3, 4, 5, 7, 9)
        if traffic_manager.ambulance is not None:
            renderer.draw_ambulance(screen, traffic_manager.ambulance, font_small, font_timer, dt)

        # 6. Four traffic signal heads (North, South, East, West) with glow & badges
        renderer.draw_all_signals(screen, signal_controller, font_small, font_timer)

        # 7. RSU smart infrastructure station with mast, antenna array, and pulsing beacon
        renderer.draw_rsu_station(screen, font_small)

        # 8. Glassmorphic HUD panels (RSU Controller, Signals, V2I Telemetry, Conflict Analysis, Event Timeline)
        renderer.draw_hud(
            screen=screen,
            signal_controller=signal_controller,
            traffic_manager=traffic_manager,
            font_title=font_title,
            font_bold=font_bold,
            font_normal=font_normal,
            font_small=font_small,
            sim_time=sim_time,
            paused=paused,
            v2i_channel=v2i_channel,
            event_logger=event_logger,
        )

        pygame.display.flip()

    pygame.quit()
    logger.info("[V2I] Simulation exited cleanly.")
    sys.exit(0)


if __name__ == "__main__":
    main()
