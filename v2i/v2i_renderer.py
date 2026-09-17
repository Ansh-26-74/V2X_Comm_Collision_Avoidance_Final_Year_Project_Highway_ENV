"""Phase 2: V2I Visual Rendering System — 4-Way PLUS Intersection, Signals, & Civilian Traffic.

Draws:
  - 4-Way PLUS (+) urban road junction with asphalt, sidewalks, and curbs
  - Lane markings, double yellow centerlines, stop lines on all 4 approaches
  - Pedestrian crosswalks and lane direction arrows
  - Central intersection yellow dashed clear box
  - 4 physical traffic signal heads (North, South, East, West) with 3 lamps each + glow
  - RSU tower mast with antenna, base unit, and pulsing V2X wireless beacon
  - Civilian vehicles with car chassis, headlights, taillights/brake-lights, badges, and speed labels
  - Glassmorphic HUD panels for Smart Intersection status, Signals, and Traffic Monitoring

This module does NOT import from any V2V module.
"""

import math
import pygame
from v2i.smart_signal import SignalState, CyclePhase
from v2i.traffic_manager import VehicleState
from v2i.event_logger import EventCategory, EventLogger

# ─────────────────────────────────────────────────────────────────────────────
# Colour Palette
# ─────────────────────────────────────────────────────────────────────────────
C_BG            = (18,  24,  36)          # Background landscape / urban ground
C_SIDEWALK      = (72,  80,  96)          # Concrete curb / sidewalk
C_ROAD          = (34,  38,  48)          # Asphalt road surface
C_ROAD_EDGE     = (90,  98, 115)          # White road border lines
C_CENTERLINE    = (235, 185,  45)         # Double yellow centerline
C_STOP_LINE     = (245, 245, 250)         # Solid white stop lines
C_CROSSWALK     = (210, 215, 225)         # White crosswalk zebra bars
C_JUNCTION_BOX  = (230, 180,  40, 160)    # Yellow dashed intersection perimeter
C_ARROW         = (180, 190, 205)         # Lane directional arrow markings

# Panel & UI Colours
C_PANEL_BG      = (12,  18,  32, 225)     # Glassmorphic dark panel fill
C_PANEL_BORDER  = (55,  85, 140)         # Panel border
C_TITLE         = (210, 230, 255)         # Header text
C_TEXT_NORMAL   = (215, 220, 230)         # Standard text
C_TEXT_DIM      = (130, 140, 160)         # Secondary / dimmed text

# Signal Lamp Colours
C_LAMP_RED_ON    = (255,  55,  55)
C_LAMP_RED_OFF   = (65,  18,  18)
C_LAMP_YEL_ON    = (255, 210,  40)
C_LAMP_YEL_OFF   = (65,  52,  15)
C_LAMP_GRN_ON    = (45,  230, 105)
C_LAMP_GRN_OFF   = (15,  55,  28)

# RSU & V2X Colours
C_RSU_MAST      = (170, 185, 205)
C_RSU_ACCENT    = (0,   210, 255)
C_V2X_BEACON    = (60,  190, 255)

# Vehicle Lights
C_HEADLIGHT     = (255, 250, 210)
C_TAILLIGHT_DIM = (160,  25,  25)
C_BRAKELIGHT_ON = (255,  45,  45)


# ─────────────────────────────────────────────────────────────────────────────
# Intersection Geometric Parameters (World & Screen Space)
# ─────────────────────────────────────────────────────────────────────────────
CX = 560             # Center X of intersection (pixels)
CY = 415             # Center Y of intersection (pixels)
ROAD_WIDTH = 130     # Total road width (2 lanes: 65px incoming, 65px outgoing)
HALF_ROAD = ROAD_WIDTH // 2  # 65 px
STOP_LINE_DIST = 75  # Distance from intersection center to stop line (pixels)
CROSSWALK_DIST = 90  # Distance from intersection center to crosswalk (pixels)
SOUTH_ROAD_EXT = 32  # Extension for South->North adjacent overtaking lane

_beacon_phase = 0.0  # Animation phase for RSU pulse


def tick_renderer(dt: float) -> None:
    """Advance renderer animation timer."""
    global _beacon_phase
    _beacon_phase = (_beacon_phase + dt * 2.0) % (2.0 * math.pi)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing: Road Geometry & Markings
# ─────────────────────────────────────────────────────────────────────────────

def draw_four_way_intersection(screen: pygame.Surface) -> None:
    """Draw the 4-way PLUS (+) intersection roads, lane markings, and stop lines."""
    w, h = screen.get_size()

    # 1. Background ground
    screen.fill(C_BG)

    # 2. Sidewalk / curb underlay (wider than road by 16px)
    curb_w = ROAD_WIDTH + 16
    half_curb = curb_w // 2
    # Vertical curb (widened on East side for South->North multi-lane corridor)
    pygame.draw.rect(screen, C_SIDEWALK, (CX - half_curb, 0, curb_w + SOUTH_ROAD_EXT, h))
    # Horizontal curb
    pygame.draw.rect(screen, C_SIDEWALK, (0, CY - half_curb, w, curb_w))
    # Corner rounded fillets
    for ox, oy in [(-half_curb, -half_curb), (half_curb + SOUTH_ROAD_EXT, -half_curb),
                   (-half_curb, half_curb), (half_curb + SOUTH_ROAD_EXT, half_curb)]:
        pygame.draw.circle(screen, C_SIDEWALK, (CX + ox, CY + oy), 18)

    # 3. Asphalt roadway
    # Vertical road (North - South, widened on East side for Lane 2)
    pygame.draw.rect(screen, C_ROAD, (CX - HALF_ROAD, 0, ROAD_WIDTH + SOUTH_ROAD_EXT, h))
    # Horizontal road (West - East)
    pygame.draw.rect(screen, C_ROAD, (0, CY - HALF_ROAD, w, ROAD_WIDTH))

    # 4. White road edge lines (solid borders)
    # North road borders
    pygame.draw.line(screen, C_ROAD_EDGE, (CX - HALF_ROAD, 0), (CX - HALF_ROAD, CY - HALF_ROAD), 2)
    pygame.draw.line(screen, C_ROAD_EDGE, (CX + HALF_ROAD + SOUTH_ROAD_EXT, 0), (CX + HALF_ROAD + SOUTH_ROAD_EXT, CY - HALF_ROAD), 2)
    # South road borders
    pygame.draw.line(screen, C_ROAD_EDGE, (CX - HALF_ROAD, CY + HALF_ROAD), (CX - HALF_ROAD, h), 2)
    pygame.draw.line(screen, C_ROAD_EDGE, (CX + HALF_ROAD + SOUTH_ROAD_EXT, CY + HALF_ROAD), (CX + HALF_ROAD + SOUTH_ROAD_EXT, h), 2)
    # West road borders
    pygame.draw.line(screen, C_ROAD_EDGE, (0, CY - HALF_ROAD), (CX - HALF_ROAD, CY - HALF_ROAD), 2)
    pygame.draw.line(screen, C_ROAD_EDGE, (0, CY + HALF_ROAD), (CX - HALF_ROAD, CY + HALF_ROAD), 2)
    # East road borders
    pygame.draw.line(screen, C_ROAD_EDGE, (CX + HALF_ROAD + SOUTH_ROAD_EXT, CY - HALF_ROAD), (w, CY - HALF_ROAD), 2)
    pygame.draw.line(screen, C_ROAD_EDGE, (CX + HALF_ROAD + SOUTH_ROAD_EXT, CY + HALF_ROAD), (w, CY + HALF_ROAD), 2)

    # 5. Centerlines (Double Yellow Lines separating opposing traffic flows)
    def draw_double_yellow(p1, p2, is_vertical=True):
        if is_vertical:
            pygame.draw.line(screen, C_CENTERLINE, (p1[0] - 2, p1[1]), (p2[0] - 2, p2[1]), 2)
            pygame.draw.line(screen, C_CENTERLINE, (p1[0] + 2, p1[1]), (p2[0] + 2, p2[1]), 2)
        else:
            pygame.draw.line(screen, C_CENTERLINE, (p1[0], p1[1] - 2), (p2[0], p2[1] - 2), 2)
            pygame.draw.line(screen, C_CENTERLINE, (p1[0], p1[1] + 2), (p2[0], p2[1] + 2), 2)

    # North approach centerline (from top of screen to crosswalk)
    draw_double_yellow((CX, 0), (CX, CY - CROSSWALK_DIST), is_vertical=True)
    # South approach centerline (from crosswalk to bottom of screen)
    draw_double_yellow((CX, CY + CROSSWALK_DIST), (CX, h), is_vertical=True)
    # West approach centerline (from left of screen to crosswalk)
    draw_double_yellow((0, CY), (CX - CROSSWALK_DIST, CY), is_vertical=False)
    # East approach centerline (from crosswalk to right of screen)
    draw_double_yellow((CX + CROSSWALK_DIST, CY), (w, CY), is_vertical=False)

    # 6. Pedestrian Crosswalks (Zebra stripes on each of the 4 approaches)
    # North crosswalk
    for x in range(CX - HALF_ROAD + 8, CX + HALF_ROAD + SOUTH_ROAD_EXT - 6, 12):
        pygame.draw.rect(screen, C_CROSSWALK, (x, CY - CROSSWALK_DIST, 7, 10))
    # South crosswalk
    for x in range(CX - HALF_ROAD + 8, CX + HALF_ROAD + SOUTH_ROAD_EXT - 6, 12):
        pygame.draw.rect(screen, C_CROSSWALK, (x, CY + CROSSWALK_DIST - 10, 7, 10))
    # West crosswalk
    for y in range(CY - HALF_ROAD + 8, CY + HALF_ROAD - 6, 12):
        pygame.draw.rect(screen, C_CROSSWALK, (CX - CROSSWALK_DIST, y, 10, 7))
    # East crosswalk
    for y in range(CY - HALF_ROAD + 8, CY + HALF_ROAD - 6, 12):
        pygame.draw.rect(screen, C_CROSSWALK, (CX + CROSSWALK_DIST - 10, y, 10, 7))

    # 7. Stop Lines (Solid white bar across incoming traffic lane)
    # North approach: incoming lane is on the WEST side (heading South)
    pygame.draw.line(screen, C_STOP_LINE, (CX - HALF_ROAD, CY - STOP_LINE_DIST), (CX, CY - STOP_LINE_DIST), 4)

    # South approach: incoming lane is on the EAST side (heading North, spanning Lane 1 and Lane 2)
    pygame.draw.line(screen, C_STOP_LINE, (CX, CY + STOP_LINE_DIST), (CX + HALF_ROAD + SOUTH_ROAD_EXT, CY + STOP_LINE_DIST), 4)

    # West approach: incoming lane is on the SOUTH side (heading East)
    pygame.draw.line(screen, C_STOP_LINE, (CX - STOP_LINE_DIST, CY), (CX - STOP_LINE_DIST, CY + HALF_ROAD), 4)

    # East approach: incoming lane is on the NORTH side (heading West)
    pygame.draw.line(screen, C_STOP_LINE, (CX + STOP_LINE_DIST, CY - HALF_ROAD), (CX + STOP_LINE_DIST, CY), 4)

    # 8. Central Intersection Box (dashed yellow clear zone)
    box_rect = pygame.Rect(CX - HALF_ROAD, CY - HALF_ROAD, ROAD_WIDTH + SOUTH_ROAD_EXT, ROAD_WIDTH)
    dash_length = 8
    # Top & bottom dashed borders
    for x in range(box_rect.left, box_rect.right, dash_length * 2):
        pygame.draw.line(screen, (220, 180, 40), (x, box_rect.top), (min(x + dash_length, box_rect.right), box_rect.top), 2)
        pygame.draw.line(screen, (220, 180, 40), (x, box_rect.bottom), (min(x + dash_length, box_rect.right), box_rect.bottom), 2)
    # Left & right dashed borders
    for y in range(box_rect.top, box_rect.bottom, dash_length * 2):
        pygame.draw.line(screen, (220, 180, 40), (box_rect.left, y), (box_rect.left, min(y + dash_length, box_rect.bottom)), 2)
        pygame.draw.line(screen, (220, 180, 40), (box_rect.right, y), (box_rect.right, min(y + dash_length, box_rect.bottom)), 2)

    # 9. Same-Direction Lane Divider (Dashed white line at x=610 separating Lane 1 from Lane 2)
    dash_h = 10
    for y in range(CY + CROSSWALK_DIST, h, 22):
        pygame.draw.line(screen, (220, 225, 235), (610, y), (610, min(y + dash_h, h)), 2)
    for y in range(0, CY - CROSSWALK_DIST, 22):
        pygame.draw.line(screen, (220, 225, 235), (610, y), (610, min(y + dash_h, CY - CROSSWALK_DIST)), 2)

    # 10. Lane Direction Arrows on Asphalt
    draw_lane_arrow(screen, (CX - 32, CY - STOP_LINE_DIST - 40), "SOUTH")
    draw_lane_arrow(screen, (592, CY + STOP_LINE_DIST + 40), "NORTH")
    draw_lane_arrow(screen, (628, CY + STOP_LINE_DIST + 40), "NORTH")
    draw_lane_arrow(screen, (CX - STOP_LINE_DIST - 40, CY + 32), "EAST")
    draw_lane_arrow(screen, (CX + STOP_LINE_DIST + 40, CY - 32), "WEST")


def draw_lane_arrow(screen: pygame.Surface, pos: tuple[int, int], direction: str) -> None:
    """Draw a lane directional arrow painted on the asphalt."""
    x, y = pos
    if direction == "SOUTH":
        pygame.draw.line(screen, C_ARROW, (x, y - 12), (x, y + 10), 3)
        pygame.draw.polygon(screen, C_ARROW, [(x - 6, y + 6), (x + 6, y + 6), (x, y + 14)])
    elif direction == "NORTH":
        pygame.draw.line(screen, C_ARROW, (x, y + 12), (x, y - 10), 3)
        pygame.draw.polygon(screen, C_ARROW, [(x - 6, y - 6), (x + 6, y - 6), (x, y - 14)])
    elif direction == "EAST":
        pygame.draw.line(screen, C_ARROW, (x - 12, y), (x + 10, y), 3)
        pygame.draw.polygon(screen, C_ARROW, [(x + 6, y - 6), (x + 6, y + 6), (x + 14, y)])
    elif direction == "WEST":
        pygame.draw.line(screen, C_ARROW, (x + 12, y), (x - 10, y), 3)
        pygame.draw.polygon(screen, C_ARROW, [(x - 6, y - 6), (x - 6, y + 6), (x - 14, y)])


# ─────────────────────────────────────────────────────────────────────────────
# Drawing: Four Traffic Signal Heads
# ─────────────────────────────────────────────────────────────────────────────

def draw_single_signal_head(
    screen: pygame.Surface,
    x: int,
    y: int,
    state: SignalState,
    label: str,
    font_label,
    font_timer,
    time_left: float,
    orientation: str = "VERTICAL",
) -> None:
    """Draw a single realistic 3-lamp traffic signal head with glow."""
    is_red = (state == SignalState.RED)
    is_yel = (state == SignalState.YELLOW)
    is_grn = (state == SignalState.GREEN)

    lamp_radius = 8
    spacing = 20

    if orientation == "VERTICAL":
        box_w, box_h = 28, 70
        bx = x - box_w // 2
        by = y - box_h // 2

        # Housing body
        pygame.draw.rect(screen, (20, 24, 30), (bx, by, box_w, box_h), border_radius=6)
        pygame.draw.rect(screen, (80, 90, 105), (bx, by, box_w, box_h), 2, border_radius=6)

        r_pos = (x, by + 14)
        y_pos = (x, by + 14 + spacing)
        g_pos = (x, by + 14 + spacing * 2)

    else:  # HORIZONTAL
        box_w, box_h = 70, 28
        bx = x - box_w // 2
        by = y - box_h // 2

        # Housing body
        pygame.draw.rect(screen, (20, 24, 30), (bx, by, box_w, box_h), border_radius=6)
        pygame.draw.rect(screen, (80, 90, 105), (bx, by, box_w, box_h), 2, border_radius=6)

        r_pos = (bx + 14, y)
        y_pos = (bx + 14 + spacing, y)
        g_pos = (bx + 14 + spacing * 2, y)

    # Red Lamp
    if is_red:
        glow_surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*C_LAMP_RED_ON, 70), (16, 16), 14)
        screen.blit(glow_surf, (r_pos[0] - 16, r_pos[1] - 16))
        pygame.draw.circle(screen, C_LAMP_RED_ON, r_pos, lamp_radius)
        pygame.draw.circle(screen, (255, 200, 200), r_pos, 3)
    else:
        pygame.draw.circle(screen, C_LAMP_RED_OFF, r_pos, lamp_radius)

    # Yellow Lamp
    if is_yel:
        glow_surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*C_LAMP_YEL_ON, 80), (16, 16), 14)
        screen.blit(glow_surf, (y_pos[0] - 16, y_pos[1] - 16))
        pygame.draw.circle(screen, C_LAMP_YEL_ON, y_pos, lamp_radius)
        pygame.draw.circle(screen, (255, 255, 200), y_pos, 3)
    else:
        pygame.draw.circle(screen, C_LAMP_YEL_OFF, y_pos, lamp_radius)

    # Green Lamp
    if is_grn:
        glow_surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*C_LAMP_GRN_ON, 80), (16, 16), 14)
        screen.blit(glow_surf, (g_pos[0] - 16, g_pos[1] - 16))
        pygame.draw.circle(screen, C_LAMP_GRN_ON, g_pos, lamp_radius)
        pygame.draw.circle(screen, (200, 255, 220), g_pos, 3)
    else:
        pygame.draw.circle(screen, C_LAMP_GRN_OFF, g_pos, lamp_radius)

    # Label text & timer tag
    tag_col = C_LAMP_RED_ON if is_red else (C_LAMP_YEL_ON if is_yel else C_LAMP_GRN_ON)
    lbl_surf = font_label.render(label, True, (240, 245, 255))
    screen.blit(lbl_surf, (x - lbl_surf.get_width() // 2, by - 16))

    sec_txt = f"{int(time_left):02d}s"
    sec_surf = font_timer.render(sec_txt, True, tag_col)
    screen.blit(sec_surf, (x - sec_surf.get_width() // 2, by + box_h + 3))


def draw_all_signals(
    screen: pygame.Surface,
    signal_controller,
    font_label,
    font_timer,
) -> None:
    """Position and render the four traffic signals at their respective approaches."""
    time_left = signal_controller.time_remaining

    # 1. NORTH Approach Signal: NW curb facing incoming southbound traffic
    draw_single_signal_head(
        screen,
        x=CX - HALF_ROAD - 25,
        y=CY - STOP_LINE_DIST - 10,
        state=signal_controller.get_signal("NORTH"),
        label="NORTH",
        font_label=font_label,
        font_timer=font_timer,
        time_left=time_left,
        orientation="VERTICAL",
    )

    # 2. SOUTH Approach Signal: SE curb facing incoming northbound traffic
    draw_single_signal_head(
        screen,
        x=CX + HALF_ROAD + SOUTH_ROAD_EXT + 22,
        y=CY + STOP_LINE_DIST + 10,
        state=signal_controller.get_signal("SOUTH"),
        label="SOUTH",
        font_label=font_label,
        font_timer=font_timer,
        time_left=time_left,
        orientation="VERTICAL",
    )

    # Phase 9: Emergency Priority Visual Emphasis near South Signal
    if (
        signal_controller.current_phase == CyclePhase.EMERGENCY_SOUTH_GREEN
        or getattr(signal_controller, "preemption_active", False)
    ):
        badge_x = CX + HALF_ROAD + SOUTH_ROAD_EXT + 36
        badge_y = CY + STOP_LINE_DIST - 10
        em_surf = font_label.render("★ EMERGENCY PRIORITY", True, (60, 240, 140))
        em_bg = pygame.Surface((em_surf.get_width() + 10, em_surf.get_height() + 6), pygame.SRCALPHA)
        em_bg.fill((15, 30, 25, 220))
        pygame.draw.rect(em_bg, (60, 240, 140), (0, 0, em_bg.get_width(), em_bg.get_height()), 1, border_radius=4)
        screen.blit(em_bg, (badge_x, badge_y))
        screen.blit(em_surf, (badge_x + 5, badge_y + 3))

    # 3. WEST Approach Signal: SW curb facing incoming eastbound traffic
    draw_single_signal_head(
        screen,
        x=CX - STOP_LINE_DIST - 15,
        y=CY + HALF_ROAD + 25,
        state=signal_controller.get_signal("WEST"),
        label="WEST",
        font_label=font_label,
        font_timer=font_timer,
        time_left=time_left,
        orientation="HORIZONTAL",
    )

    # 4. EAST Approach Signal: NE curb facing incoming westbound traffic
    draw_single_signal_head(
        screen,
        x=CX + STOP_LINE_DIST + 15,
        y=CY - HALF_ROAD - 25,
        state=signal_controller.get_signal("EAST"),
        label="EAST",
        font_label=font_label,
        font_timer=font_timer,
        time_left=time_left,
        orientation="HORIZONTAL",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Drawing: Civilian Vehicles
# ─────────────────────────────────────────────────────────────────────────────

def draw_civilian_vehicles(
    screen: pygame.Surface,
    vehicles: list,
    font_id,
    font_speed,
) -> None:
    """Draw civilian vehicles with headlights, taillights, roof cabin, and speed."""
    for v in vehicles:
        # Compute bounding rectangle based on heading orientation
        if v.approach in ("NORTH", "SOUTH"):
            w_box, h_box = int(v.width), int(v.length)
        else:
            w_box, h_box = int(v.length), int(v.width)

        rect_x = int(v.x - w_box / 2)
        rect_y = int(v.y - h_box / 2)

        # 1. Vehicle Chassis (main body)
        veh_surf = pygame.Surface((w_box, h_box), pygame.SRCALPHA)
        pygame.draw.rect(veh_surf, v.color, (0, 0, w_box, h_box), border_radius=5)
        pygame.draw.rect(veh_surf, (15, 20, 30), (0, 0, w_box, h_box), 1, border_radius=5)

        # 2. Cabin / Windshield Tint
        if v.approach == "NORTH":  # Moving South (+Y)
            # Front windshield at bottom, rear window at top
            pygame.draw.rect(veh_surf, (25, 35, 50), (2, int(h_box * 0.45), w_box - 4, int(h_box * 0.42)), border_radius=3)
            pygame.draw.rect(veh_surf, (15, 20, 30), (3, int(h_box * 0.18), w_box - 6, int(h_box * 0.22)), border_radius=2)
        elif v.approach == "SOUTH":  # Moving North (-Y)
            # Front windshield at top, rear window at bottom
            pygame.draw.rect(veh_surf, (25, 35, 50), (2, int(h_box * 0.13), w_box - 4, int(h_box * 0.42)), border_radius=3)
            pygame.draw.rect(veh_surf, (15, 20, 30), (3, int(h_box * 0.60), w_box - 6, int(h_box * 0.22)), border_radius=2)
        elif v.approach == "WEST":  # Moving East (+X)
            # Front windshield at right, rear window at left
            pygame.draw.rect(veh_surf, (25, 35, 50), (int(w_box * 0.45), 2, int(w_box * 0.42), h_box - 4), border_radius=3)
            pygame.draw.rect(veh_surf, (15, 20, 30), (int(w_box * 0.18), 3, int(w_box * 0.22), h_box - 6), border_radius=2)
        elif v.approach == "EAST":  # Moving West (-X)
            # Front windshield at left, rear window at right
            pygame.draw.rect(veh_surf, (25, 35, 50), (int(w_box * 0.13), 2, int(w_box * 0.42), h_box - 4), border_radius=3)
            pygame.draw.rect(veh_surf, (15, 20, 30), (int(w_box * 0.60), 3, int(w_box * 0.22), h_box - 6), border_radius=2)

        screen.blit(veh_surf, (rect_x, rect_y))

        # 3. Headlights and Taillights (with braking halos)
        brake_active = v.braking or (v.speed < 0.5)

        if v.approach == "NORTH":  # Front is at Y bottom, Rear is at Y top
            # Headlights (bottom)
            pygame.draw.circle(screen, C_HEADLIGHT, (rect_x + 3, rect_y + h_box), 2)
            pygame.draw.circle(screen, C_HEADLIGHT, (rect_x + w_box - 3, rect_y + h_box), 2)
            # Taillights (top)
            tail_col = C_BRAKELIGHT_ON if brake_active else C_TAILLIGHT_DIM
            pygame.draw.circle(screen, tail_col, (rect_x + 3, rect_y), 3 if brake_active else 2)
            pygame.draw.circle(screen, tail_col, (rect_x + w_box - 3, rect_y), 3 if brake_active else 2)

        elif v.approach == "SOUTH":  # Front is at Y top, Rear is at Y bottom
            # Headlights (top)
            pygame.draw.circle(screen, C_HEADLIGHT, (rect_x + 3, rect_y), 2)
            pygame.draw.circle(screen, C_HEADLIGHT, (rect_x + w_box - 3, rect_y), 2)
            # Taillights (bottom)
            tail_col = C_BRAKELIGHT_ON if brake_active else C_TAILLIGHT_DIM
            pygame.draw.circle(screen, tail_col, (rect_x + 3, rect_y + h_box), 3 if brake_active else 2)
            pygame.draw.circle(screen, tail_col, (rect_x + w_box - 3, rect_y + h_box), 3 if brake_active else 2)

        elif v.approach == "WEST":  # Front is at X right, Rear is at X left
            # Headlights (right)
            pygame.draw.circle(screen, C_HEADLIGHT, (rect_x + w_box, rect_y + 3), 2)
            pygame.draw.circle(screen, C_HEADLIGHT, (rect_x + w_box, rect_y + h_box - 3), 2)
            # Taillights (left)
            tail_col = C_BRAKELIGHT_ON if brake_active else C_TAILLIGHT_DIM
            pygame.draw.circle(screen, tail_col, (rect_x, rect_y + 3), 3 if brake_active else 2)
            pygame.draw.circle(screen, tail_col, (rect_x, rect_y + h_box - 3), 3 if brake_active else 2)

        elif v.approach == "EAST":  # Front is at X left, Rear is at X right
            # Headlights (left)
            pygame.draw.circle(screen, C_HEADLIGHT, (rect_x, rect_y + 3), 2)
            pygame.draw.circle(screen, C_HEADLIGHT, (rect_x, rect_y + h_box - 3), 2)
            # Taillights (right)
            tail_col = C_BRAKELIGHT_ON if brake_active else C_TAILLIGHT_DIM
            pygame.draw.circle(screen, tail_col, (rect_x + w_box, rect_y + 3), 3 if brake_active else 2)
            pygame.draw.circle(screen, tail_col, (rect_x + w_box, rect_y + h_box - 3), 3 if brake_active else 2)

        # 4. Vehicle ID & State Tag
        id_txt = font_id.render(v.vehicle_id, True, (245, 245, 255))
        screen.blit(id_txt, (v.x - id_txt.get_width() // 2, v.y - id_txt.get_height() // 2))

        # Speed tag beside vehicle
        spd_kmh = int(v.speed * 0.36)  # Scale ~0.36 to km/h
        if v.speed == 0.0:
            spd_str = "STOP"
            spd_col = C_LAMP_RED_ON
        elif v.braking:
            spd_str = f"{spd_kmh} ↓"
            spd_col = C_LAMP_YEL_ON
        else:
            spd_str = f"{spd_kmh}"
            spd_col = C_LAMP_GRN_ON

        spd_surf = font_speed.render(spd_str, True, spd_col)
        # Position badge offset to side of vehicle
        if v.approach in ("NORTH", "SOUTH"):
            screen.blit(spd_surf, (rect_x + w_box + 4, v.y - 6))
        else:
            screen.blit(spd_surf, (v.x - spd_surf.get_width() // 2, rect_y - 14))


# ─────────────────────────────────────────────────────────────────────────────
# Drawing: Emergency Vehicle AMB-01 (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

def draw_ambulance(
    screen: pygame.Surface,
    ambulance,
    font_id,
    font_speed,
    dt: float = 1.0 / 60.0,
) -> None:
    """Draw emergency vehicle AMB-01 with distinct ambulance livery and flashing lightbars."""
    if ambulance is None:
        return

    v = ambulance
    w_box, h_box = int(v.width), int(v.length)  # 20px wide, 44px long (South -> North)
    rect_x = int(v.x - w_box / 2)
    rect_y = int(v.y - h_box / 2)

    # 1. Outer Glow / Emergency Aura when strobe is active
    glow_surf = pygame.Surface((w_box + 24, h_box + 24), pygame.SRCALPHA)
    aura_col = (255, 50, 50, 45) if v.strobe_left_on else (40, 120, 255, 45)
    pygame.draw.ellipse(glow_surf, aura_col, (0, 0, w_box + 24, h_box + 24))
    screen.blit(glow_surf, (rect_x - 12, rect_y - 12))

    # 2. Main Chassis (Crisp Ambulance White with subtle metallic tint)
    veh_surf = pygame.Surface((w_box, h_box), pygame.SRCALPHA)
    pygame.draw.rect(veh_surf, (250, 250, 252), (0, 0, w_box, h_box), border_radius=6)
    pygame.draw.rect(veh_surf, (30, 35, 45), (0, 0, w_box, h_box), 1, border_radius=6)

    # Red Emergency Side Stripes
    pygame.draw.rect(veh_surf, (220, 35, 45), (0, 6, 3, h_box - 12))
    pygame.draw.rect(veh_surf, (220, 35, 45), (w_box - 3, 6, 3, h_box - 12))

    # Rear Bumper Warning Chevrons (Moving North, so rear is at bottom of box)
    pygame.draw.rect(veh_surf, (255, 200, 30), (2, h_box - 4, w_box - 4, 3))
    pygame.draw.rect(veh_surf, (220, 30, 30), (5, h_box - 4, 3, 3))
    pygame.draw.rect(veh_surf, (220, 30, 30), (12, h_box - 4, 3, 3))

    # 3. Windshield (Front is at top of box when moving North)
    pygame.draw.rect(veh_surf, (25, 35, 50), (3, 6, w_box - 6, 7), border_radius=2)
    # Rear Windows
    pygame.draw.rect(veh_surf, (30, 40, 55), (4, h_box - 10, 4, 5), border_radius=1)
    pygame.draw.rect(veh_surf, (30, 40, 55), (w_box - 8, h_box - 10, 4, 5), border_radius=1)

    # 4. Red Cross Emblem on Roof
    cx_box = w_box // 2
    cy_box = h_box // 2 + 3
    # Red cross arms
    pygame.draw.rect(veh_surf, (225, 25, 35), (cx_box - 2, cy_box - 5, 4, 10))
    pygame.draw.rect(veh_surf, (225, 25, 35), (cx_box - 5, cy_box - 2, 10, 4))

    # 5. Dual Flashing Lightbar on Front Roof
    bar_y = 15
    # Lightbar housing
    pygame.draw.rect(veh_surf, (20, 25, 35), (cx_box - 8, bar_y, 16, 5), border_radius=2)
    # Left Strobe (Red)
    left_col = (255, 30, 30) if v.strobe_left_on else (90, 15, 15)
    pygame.draw.rect(veh_surf, left_col, (cx_box - 7, bar_y + 1, 6, 3), border_radius=1)
    # Right Strobe (Blue)
    right_col = (30, 140, 255) if not v.strobe_left_on else (15, 35, 90)
    pygame.draw.rect(veh_surf, right_col, (cx_box + 1, bar_y + 1, 6, 3), border_radius=1)

    # Blit chassis
    screen.blit(veh_surf, (rect_x, rect_y))

    # 6. Headlights & Brake lights in World Space
    # Front Headlights (Facing North: at top of vehicle rect_y)
    hl_col = C_HEADLIGHT
    screen.fill(hl_col, (rect_x + 2, rect_y - 2, 4, 2))
    screen.fill(hl_col, (rect_x + w_box - 6, rect_y - 2, 4, 2))

    # Rear Taillights / Brake lights (Facing South: at bottom of vehicle rect_y + h_box)
    tl_col = C_BRAKELIGHT_ON if (v.braking or v.speed == 0.0) else C_TAILLIGHT_DIM
    screen.fill(tl_col, (rect_x + 2, rect_y + h_box, 4, 2))
    screen.fill(tl_col, (rect_x + w_box - 6, rect_y + h_box, 4, 2))

    # 7. Labels & Badges
    # ID Badge above vehicle
    id_txt = font_id.render("AMB-01", True, (255, 235, 80))
    screen.blit(id_txt, (v.x - id_txt.get_width() // 2, rect_y - 15))

    # Speed Tag / State Indicator beside vehicle
    spd_kmh = int(v.speed * 0.36)
    if getattr(v, "cleared", False):
        spd_str = "CLEARED"
        spd_col = (60, 225, 120)
    elif getattr(v, "in_intersection", False):
        spd_str = f"{spd_kmh} km/h [CROSSING]"
        spd_col = (60, 225, 120)
    elif getattr(v, "is_overtaking", False) or getattr(v, "state", None) == VehicleState.OVERTAKING:
        spd_str = f"{spd_kmh} km/h [OVERTAKING]"
        spd_col = (255, 185, 45)
    elif getattr(v, "is_authorized", False):
        spd_str = f"{spd_kmh} km/h [PRIORITY GO]"
        spd_col = (80, 220, 255)
    elif v.speed == 0.0:
        spd_str = "STOP [WAIT V2I]"
        spd_col = C_LAMP_RED_ON
    elif v.braking:
        spd_str = f"{spd_kmh} ↓ BRAKE"
        spd_col = C_LAMP_YEL_ON
    else:
        spd_str = f"{spd_kmh} km/h"
        spd_col = (60, 220, 255)

    spd_surf = font_speed.render(spd_str, True, spd_col)
    screen.blit(spd_surf, (rect_x + w_box + 6, v.y - 7))

    # Phase 9: Overtaking Target Lane Projection Indicator
    if getattr(v, "is_overtaking", False) or getattr(v, "state", None) == VehicleState.OVERTAKING:
        target_x = getattr(v, "target_lane_x", 628.0)
        # Subtle directional guideline to target passing corridor
        pygame.draw.line(
            screen,
            (255, 195, 45, 180),
            (int(v.x), int(v.y - h_box / 2)),
            (int(target_x), int(v.y - h_box / 2 - 35)),
            2,
        )
        pygame.draw.circle(screen, (255, 215, 60), (int(target_x), int(v.y - h_box / 2 - 35)), 4)
        ot_txt = font_id.render("TARGET: PASSING LANE", True, (255, 205, 50))
        screen.blit(ot_txt, (int(target_x - ot_txt.get_width() // 2), int(v.y - h_box / 2 - 50)))


# ─────────────────────────────────────────────────────────────────────────────
# Drawing: RSU / Smart Infrastructure Station
# ─────────────────────────────────────────────────────────────────────────────

RSU_X = CX + HALF_ROAD + 110
RSU_Y = CY - HALF_ROAD - 95
RSU_POS = (RSU_X, RSU_Y)


def draw_rsu_station(screen: pygame.Surface, font_small) -> None:
    """Draw the physical RSU (Roadside Unit) mast, antenna, and status beacon."""
    rsu_x, rsu_y = RSU_POS

    # Ground concrete foundation
    pygame.draw.rect(screen, (50, 58, 70), (rsu_x - 18, rsu_y + 35, 36, 12), border_radius=3)

    # Vertical mast pole
    pygame.draw.line(screen, C_RSU_MAST, (rsu_x, rsu_y - 25), (rsu_x, rsu_y + 35), 4)

    # Equipment housing box
    pygame.draw.rect(screen, (35, 45, 60), (rsu_x - 12, rsu_y + 10, 24, 20), border_radius=4)
    pygame.draw.rect(screen, C_PANEL_BORDER, (rsu_x - 12, rsu_y + 10, 24, 20), 1, border_radius=4)

    # Top crossbar & antenna radomes
    pygame.draw.line(screen, C_RSU_MAST, (rsu_x - 16, rsu_y - 25), (rsu_x + 16, rsu_y - 25), 3)
    # Left antenna
    pygame.draw.line(screen, (100, 200, 255), (rsu_x - 14, rsu_y - 25), (rsu_x - 14, rsu_y - 40), 2)
    # Right antenna
    pygame.draw.line(screen, (100, 200, 255), (rsu_x + 14, rsu_y - 25), (rsu_x + 14, rsu_y - 40), 2)
    # Center directional DSRC transceiver dish
    pygame.draw.polygon(screen, (220, 230, 245), [
        (rsu_x - 7, rsu_y - 25), (rsu_x + 7, rsu_y - 25),
        (rsu_x + 10, rsu_y - 32), (rsu_x - 10, rsu_y - 32)
    ])

    # Pulsing V2X Communication Beacon Rings
    pulse_radius = 16 + int(10 * math.sin(_beacon_phase))
    pulse_alpha = int(120 + 80 * math.sin(_beacon_phase))
    glow_surf = pygame.Surface((80, 80), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (*C_V2X_BEACON, pulse_alpha // 3), (40, 40), pulse_radius)
    pygame.draw.circle(glow_surf, (*C_V2X_BEACON, pulse_alpha), (40, 40), pulse_radius, 2)
    pygame.draw.circle(glow_surf, (255, 255, 255, 220), (40, 40), 4)
    screen.blit(glow_surf, (rsu_x - 40, rsu_y - 35 - 40))

    # Text tag for RSU
    lbl = font_small.render("📡 RSU-01 [SMART CONTROLLER]", True, (80, 215, 255))
    screen.blit(lbl, (rsu_x - lbl.get_width() // 2, rsu_y + 52))

    sub_lbl = font_small.render("V2X 5.9 GHz DSRC / C-V2X", True, C_TEXT_DIM)
    screen.blit(sub_lbl, (rsu_x - sub_lbl.get_width() // 2, rsu_y + 70))


def draw_v2i_communication(
    screen: pygame.Surface,
    v2i_channel,
    ambulance=None,
    font_small=None,
    signal_controller=None,
) -> None:
    """Render V2I wireless communication range, active link beam, and in-flight packets."""
    if v2i_channel is None:
        return

    w, h = screen.get_size()
    rsu_x, rsu_y = RSU_POS

    # 1. Subtle V2I coverage zone circle around RSU
    range_radius = int(v2i_channel.v2i_range)
    range_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.circle(range_surf, (0, 190, 255, 14), (rsu_x, rsu_y), range_radius)
    pygame.draw.circle(range_surf, (0, 200, 255, 45), (rsu_x, rsu_y), range_radius, 1)

    # Range perimeter label
    if font_small:
        lbl_r = font_small.render(f"V2I RANGE ({range_radius}px)", True, (0, 190, 255, 120))
        range_surf.blit(lbl_r, (rsu_x - lbl_r.get_width() // 2, rsu_y + range_radius + 4))

    # 2. Wireless Link Beam between Ambulance and RSU when in range
    if ambulance is not None:
        dx = rsu_x - ambulance.x
        dy = rsu_y - ambulance.y
        dist = math.hypot(dx, dy)
        if dist <= v2i_channel.v2i_range:
            # Pulsing wireless connection beam
            beam_alpha = int(140 + 75 * math.sin(_beacon_phase * 3.0))
            pygame.draw.line(
                range_surf,
                (0, 230, 255, beam_alpha),
                (int(ambulance.x), int(ambulance.y)),
                (rsu_x, rsu_y),
                2,
            )
            # Glowing link endpoints
            pygame.draw.circle(range_surf, (0, 240, 255, beam_alpha), (int(ambulance.x), int(ambulance.y)), 6, 1)
            pygame.draw.circle(range_surf, (0, 240, 255, beam_alpha), (rsu_x, rsu_y), 6, 1)

    screen.blit(range_surf, (0, 0))

    # 3. Render animated flying packets
    for anim in v2i_channel.packet_animations:
        prog = max(0.0, min(1.0, anim.progress))
        cur_x = int(anim.start_pos[0] + (anim.end_pos[0] - anim.start_pos[0]) * prog)
        cur_y = int(anim.start_pos[1] + (anim.end_pos[1] - anim.start_pos[1]) * prog)

        # Draw glowing packet dot
        pkt_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(pkt_surf, (255, 215, 60, 100), (15, 15), 10)
        pygame.draw.circle(pkt_surf, (255, 235, 100, 220), (15, 15), 5)
        pygame.draw.circle(pkt_surf, (255, 255, 255), (15, 15), 2)
        screen.blit(pkt_surf, (cur_x - 15, cur_y - 15))

        # Floating packet badge
        if font_small:
            badge_surf = font_small.render("EMERGENCY_REQUEST", True, (255, 240, 150))
            badge_bg = pygame.Surface((badge_surf.get_width() + 8, badge_surf.get_height() + 4), pygame.SRCALPHA)
            badge_bg.fill((15, 25, 40, 200))
            pygame.draw.rect(badge_bg, (255, 200, 50), (0, 0, badge_bg.get_width(), badge_bg.get_height()), 1, border_radius=3)
            screen.blit(badge_bg, (cur_x + 10, cur_y - 8))
            screen.blit(badge_surf, (cur_x + 14, cur_y - 6))

    # 4. Bidirectional Visualization: RSU -> Traffic Signal Preemption Command
    if signal_controller is not None and (
        signal_controller.current_phase == CyclePhase.EMERGENCY_SOUTH_GREEN
        or getattr(signal_controller, "preemption_requested", False)
        or getattr(signal_controller, "preemption_active", False)
        or getattr(signal_controller, "emergency_terminating", False)
    ):
        pygame.draw.line(screen, (255, 205, 50), (rsu_x - 12, rsu_y + 18), (CX, CY - 10), 2)
        pygame.draw.circle(screen, (255, 220, 70), (CX, CY - 10), 5)
        if font_small:
            cmd_lbl = font_small.render("⚡ PREEMPTION COMMAND", True, (255, 220, 70))
            mid_x = (rsu_x + CX) // 2
            mid_y = (rsu_y + CY - 10) // 2
            cmd_bg = pygame.Surface((cmd_lbl.get_width() + 8, cmd_lbl.get_height() + 4), pygame.SRCALPHA)
            cmd_bg.fill((20, 25, 35, 220))
            pygame.draw.rect(cmd_bg, (255, 190, 40), (0, 0, cmd_bg.get_width(), cmd_bg.get_height()), 1, border_radius=3)
            screen.blit(cmd_bg, (mid_x - cmd_bg.get_width() // 2, mid_y - cmd_bg.get_height() // 2))
            screen.blit(cmd_lbl, (mid_x - cmd_lbl.get_width() // 2 + 4, mid_y - cmd_lbl.get_height() // 2 + 2))


def draw_conflict_zone(
    screen: pygame.Surface,
    analysis=None,
    font_small=None,
) -> None:
    """Render the central intersection conflict zone overlay based on real-time analysis."""
    if analysis is None:
        return

    cz_x = CX - HALF_ROAD - 6
    cz_y = CY - HALF_ROAD - 6
    cz_w = ROAD_WIDTH + 12
    cz_h = ROAD_WIDTH + 12

    # Status-based visual appearance
    if analysis.intersection_occupied:
        fill_col = (255, 60, 60, 42)
        border_col = (255, 75, 75)
        tag_text = "⚠️ CONFLICT ZONE: OCCUPIED"
        tag_col = (255, 90, 90)
    elif analysis.conflict_detected:
        fill_col = (255, 185, 45, 30)
        border_col = (255, 200, 50)
        tag_text = "⚡ CONFLICT ZONE: APPROACHING TRAFFIC"
        tag_col = (255, 215, 60)
    else:
        fill_col = (40, 220, 120, 22)
        border_col = (50, 235, 130)
        tag_text = "✓ CONFLICT ZONE: CLEAR"
        tag_col = (60, 225, 130)

    # Semi-transparent conflict zone fill & dashed/bordered perimeter
    cz_surf = pygame.Surface((cz_w, cz_h), pygame.SRCALPHA)
    cz_surf.fill(fill_col)
    pygame.draw.rect(cz_surf, border_col, (0, 0, cz_w, cz_h), 2, border_radius=4)
    # Corner brackets for HUD aesthetic
    bracket_len = 10
    pygame.draw.line(cz_surf, border_col, (0, 0), (bracket_len, 0), 3)
    pygame.draw.line(cz_surf, border_col, (0, 0), (0, bracket_len), 3)
    pygame.draw.line(cz_surf, border_col, (cz_w - 1, 0), (cz_w - bracket_len, 0), 3)
    pygame.draw.line(cz_surf, border_col, (cz_w - 1, 0), (cz_w - 1, bracket_len), 3)
    pygame.draw.line(cz_surf, border_col, (0, cz_h - 1), (bracket_len, cz_h - 1), 3)
    pygame.draw.line(cz_surf, border_col, (0, cz_h - 1), (0, cz_h - bracket_len), 3)
    pygame.draw.line(cz_surf, border_col, (cz_w - 1, cz_h - 1), (cz_w - bracket_len, cz_h - 1), 3)
    pygame.draw.line(cz_surf, border_col, (cz_w - 1, cz_h - 1), (cz_w - 1, cz_h - bracket_len), 3)

    screen.blit(cz_surf, (cz_x, cz_y))

    # Floating indicator badge
    if font_small:
        lbl_surf = font_small.render(tag_text, True, tag_col)
        bg_w, bg_h = lbl_surf.get_width() + 10, lbl_surf.get_height() + 4
        lbl_bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
        lbl_bg.fill((12, 18, 30, 215))
        pygame.draw.rect(lbl_bg, border_col, (0, 0, bg_w, bg_h), 1, border_radius=3)
        lbl_x = cz_x + cz_w // 2 - bg_w // 2
        lbl_y = cz_y - bg_h - 4
        screen.blit(lbl_bg, (lbl_x, lbl_y))
        screen.blit(lbl_surf, (lbl_x + 5, lbl_y + 2))


def draw_vehicle_conflict_highlights(
    screen: pygame.Surface,
    civilian_vehicles: list,
    font_small=None,
) -> None:
    """Render subtle visual warning halos around vehicles classified in conflict."""
    for v in civilian_vehicles:
        status = getattr(v, "conflict_status", None)
        if not status or status in ("SAFE", "CLEARED"):
            continue

        if v.approach in ("NORTH", "SOUTH"):
            w_box, h_box = int(v.width) + 8, int(v.length) + 8
        else:
            w_box, h_box = int(v.length) + 8, int(v.width) + 8

        rect_x = int(v.x - w_box / 2)
        rect_y = int(v.y - h_box / 2)

        if status == "IN_CONFLICT_ZONE":
            ring_col = (255, 65, 65)
            badge_txt = "CONFLICT"
            bg_col = (255, 50, 50, 40)
        else:  # APPROACHING_CONFLICT
            ring_col = (255, 195, 45)
            badge_txt = "APPROACHING"
            bg_col = (255, 180, 40, 30)

        # Highlight perimeter
        h_surf = pygame.Surface((w_box, h_box), pygame.SRCALPHA)
        h_surf.fill(bg_col)
        pygame.draw.rect(h_surf, ring_col, (0, 0, w_box, h_box), 1, border_radius=4)
        screen.blit(h_surf, (rect_x, rect_y))

        # Mini label badge
        if font_small:
            badge_surf = font_small.render(badge_txt, True, ring_col)
            screen.blit(badge_surf, (rect_x + w_box + 4, rect_y + 2))


# ─────────────────────────────────────────────────────────────────────────────
# Drawing: HUD & Telemetry Panels
# ─────────────────────────────────────────────────────────────────────────────

def draw_event_timeline(
    screen: pygame.Surface,
    event_logger,
    font_bold,
    font_normal,
    font_small,
    px: int,
    py: int,
    panel_w: int,
    panel_h: int,
) -> None:
    """Render the Phase 9 chronological event timeline audit log."""
    p_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    p_surf.fill(C_PANEL_BG)
    screen.blit(p_surf, (px, py))
    pygame.draw.rect(screen, C_PANEL_BORDER, (px, py, panel_w, panel_h), 2, border_radius=8)

    # Header
    title = font_bold.render("V2I EVENT TIMELINE (AUDIT LOG)", True, (255, 195, 60))
    screen.blit(title, (px + 14, py + 10))
    pygame.draw.line(screen, C_PANEL_BORDER, (px + 10, py + 30), (px + panel_w - 10, py + 30), 1)

    events = event_logger.get_events() if event_logger else []
    if not events:
        screen.blit(font_normal.render("Awaiting system state transitions...", True, C_TEXT_DIM), (px + 14, py + 42))
        return

    # Render recent events (newest at bottom, showing up to 8-9 visible events)
    visible_events = events[-9:]
    for i, evt in enumerate(visible_events):
        ey = py + 34 + i * 19

        # Category tag and color mapping
        if evt.category == EventCategory.COMMUNICATION:
            cat_col = (80, 215, 255)
            badge = "COMM"
        elif evt.category == EventCategory.EMERGENCY:
            cat_col = (255, 100, 80)
            badge = "EMRG"
        elif evt.category == EventCategory.SAFETY:
            cat_col = (60, 230, 130)
            badge = "SAFE"
        elif evt.category == EventCategory.SIGNAL:
            cat_col = (255, 215, 60)
            badge = "SIGN"
        else:
            cat_col = (210, 220, 235)
            badge = "TRAF"

        # Timestamp
        ts_str = f"{evt.timestamp:04.1f}s"
        screen.blit(font_small.render(ts_str, True, (160, 175, 195)), (px + 12, ey))

        # Category badge
        badge_surf = font_small.render(f"[{badge}]", True, cat_col)
        screen.blit(badge_surf, (px + 56, ey))

        # Message (truncated if exceeding panel width)
        msg_str = evt.message
        max_w = panel_w - 112
        if font_small.size(msg_str)[0] > max_w:
            while len(msg_str) > 5 and font_small.size(msg_str + "..")[0] > max_w:
                msg_str = msg_str[:-2]
            msg_str += ".."

        screen.blit(font_small.render(msg_str, True, (230, 235, 245)), (px + 102, ey))


def draw_hud(
    screen: pygame.Surface,
    signal_controller,
    traffic_manager=None,
    font_title=None,
    font_bold=None,
    font_normal=None,
    sim_time: float = 0.0,
    paused: bool = False,
    v2i_channel=None,
    event_logger=None,
    font_small=None,
) -> None:
    """Draw the smart intersection controller HUD and telemetry panels."""
    w, h = screen.get_size()
    if font_small is None:
        font_small = pygame.font.SysFont("consolas", 12, bold=True)

    # 1. Top Title Banner
    title_surf = pygame.Surface((w - 40, 40), pygame.SRCALPHA)
    title_surf.fill(C_PANEL_BG)
    screen.blit(title_surf, (20, 10))
    pygame.draw.rect(screen, C_PANEL_BORDER, (20, 10, w - 40, 40), 1, border_radius=6)

    title_text = font_title.render("V2I SMART INTERSECTION — OBSERVABILITY & EVENT TIMELINE (PHASE 9)", True, C_TITLE)
    screen.blit(title_text, (35, 18))

    pause_badge = " [PAUSED]" if paused else " [RUNNING]"
    time_text = font_bold.render(f"SIM TIME: {sim_time:05.1f}s {pause_badge}", True, (255, 215, 60) if paused else (60, 220, 120))
    screen.blit(time_text, (w - time_text.get_width() - 35, 19))

    # 2. Left Top Panel: RSU Traffic Signal Controller Status
    panel_w, panel_h = 360, 275
    px, py = 20, 55

    p_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    p_surf.fill(C_PANEL_BG)
    screen.blit(p_surf, (px, py))
    pygame.draw.rect(screen, C_PANEL_BORDER, (px, py, panel_w, panel_h), 2, border_radius=8)

    # Panel Title
    rsu_title = font_bold.render("SMART INTERSECTION / RSU-01", True, C_TITLE)
    screen.blit(rsu_title, (px + 14, py + 12))
    pygame.draw.line(screen, C_PANEL_BORDER, (px + 10, py + 34), (px + panel_w - 10, py + 34), 1)

    # Phase info
    curr_phase = signal_controller.current_phase
    phase_name = curr_phase.value.replace("_", " ")
    time_left = signal_controller.time_remaining
    total_dur = signal_controller.phase_duration

    # Preemption status determination
    if curr_phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
        preempt_badge = "ACTIVE (SOUTH GREEN)"
        preempt_col = (60, 220, 120)
        mode_str = "Mode: EMERGENCY PREEMPTION (SOUTH PRIORITY)"
    elif curr_phase == CyclePhase.EMERGENCY_TERMINATING:
        preempt_badge = "TERMINATING (SOUTH YELLOW)"
        preempt_col = (255, 215, 60)
        mode_str = "Mode: EMERGENCY TERMINATING (YELLOW CLEARANCE)"
    elif curr_phase == CyclePhase.RECOVERY_ALL_RED:
        preempt_badge = "ALL-RED CLEARANCE"
        preempt_col = (255, 80, 80)
        mode_str = "Mode: POST-EMERGENCY ALL-RED CLEARANCE"
    elif curr_phase == CyclePhase.PREEMPTION_ALL_RED:
        preempt_badge = "ALL-RED CLEARANCE"
        preempt_col = (255, 80, 80)
        mode_str = "Mode: ALL-RED SAFETY GATE CLEARANCE"
    elif curr_phase == CyclePhase.PREEMPTION_YELLOW:
        preempt_badge = "CLEARING (YELLOW)"
        preempt_col = (255, 215, 60)
        mode_str = "Mode: PREEMPTION YELLOW CLEARANCE"
    elif getattr(signal_controller, "preemption_requested", False):
        preempt_badge = "REQUESTED"
        preempt_col = (255, 215, 60)
        mode_str = "Mode: PREEMPTION REQUESTED (QUEUED)"
    else:
        preempt_badge = "NORMAL OPERATION"
        preempt_col = C_TEXT_DIM
        mode_str = "Mode: NORMAL TIMED CYCLE (COORDINATED)"

    priority_status = getattr(signal_controller, "rsu_priority_status", "INACTIVE")
    priority_col = (60, 220, 120) if priority_status == "ACTIVE" else (
        (255, 215, 60) if priority_status == "TERMINATING" else C_TEXT_DIM
    )

    lines = [
        ("Intersection ID :", signal_controller.intersection_id, C_TEXT_NORMAL),
        ("Preemption State:", preempt_badge, preempt_col),
        ("Emergency Priority:", priority_status, priority_col),
        ("Current Phase   :", phase_name, (255, 215, 60) if "YELLOW" in phase_name else ((60, 220, 120) if "GREEN" in phase_name else (255, 80, 80))),
        ("Phase Timer     :", f"{time_left:04.1f}s / {total_dur:04.1f}s", (240, 240, 240)),
        ("Cycle Count     :", f"Cycle #{signal_controller.cycle_count}", C_TEXT_DIM),
    ]

    for i, (label, val, col) in enumerate(lines):
        ly = py + 40 + i * 21
        screen.blit(font_normal.render(label, True, C_TEXT_DIM), (px + 14, ly))
        screen.blit(font_bold.render(val, True, col), (px + 160, ly))

    # Progress bar for active phase
    progress = 1.0 - (time_left / max(total_dur, 0.1))
    bar_x, bar_y, bar_w, bar_h = px + 14, py + 172, panel_w - 28, 10
    pygame.draw.rect(screen, (30, 40, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=5)
    bar_fill_col = (255, 210, 40) if "YELLOW" in phase_name else (60, 220, 120)
    pygame.draw.rect(screen, bar_fill_col, (bar_x, bar_y, int(bar_w * progress), bar_h), border_radius=5)
    pygame.draw.rect(screen, C_PANEL_BORDER, (bar_x, bar_y, bar_w, bar_h), 1, border_radius=5)

    # Safety Invariant Status badge
    safe_badge = "✓ MUTEX ENFORCED (NO CONFLICTING GREEN)"
    safe_surf = font_normal.render(safe_badge, True, (60, 225, 130))
    screen.blit(safe_surf, (px + 14, py + 196))

    mode_txt = font_normal.render(mode_str, True, (255, 215, 60) if getattr(signal_controller, "preemption_active", False) or getattr(signal_controller, "preemption_clearing", False) or getattr(signal_controller, "emergency_terminating", False) else C_TEXT_DIM)
    screen.blit(mode_txt, (px + 14, py + 220))

    # 3. Bottom-Left Panel: V2I Wireless Communication & Telemetry (Phase 4)
    left_margin = 20
    bottom_margin = 50
    v_panel_w = 360
    v_panel_h = 210
    v_px = left_margin
    v_py = h - v_panel_h - bottom_margin

    v_surf = pygame.Surface((v_panel_w, v_panel_h), pygame.SRCALPHA)
    v_surf.fill(C_PANEL_BG)
    screen.blit(v_surf, (v_px, v_py))
    pygame.draw.rect(screen, C_PANEL_BORDER, (v_px, v_py, v_panel_w, v_panel_h), 2, border_radius=8)

    v2i_title = font_bold.render("V2I WIRELESS COMM (DSRC / C-V2X)", True, (80, 215, 255))
    screen.blit(v2i_title, (v_px + 14, v_py + 10))
    pygame.draw.line(screen, C_PANEL_BORDER, (v_px + 10, v_py + 30), (v_px + v_panel_w - 10, v_py + 30), 1)

    comm_st = v2i_channel.comm_state if v2i_channel else "OFFLINE"
    st_col = (60, 220, 120) if comm_st in ("DELIVERED", "IN_RANGE", "TRANSMITTED") else (
        (255, 215, 60) if comm_st == "OUT_OF_RANGE" else (255, 80, 80)
    )

    req_received = signal_controller.emergency_request_received
    latest_req = signal_controller.latest_emergency_request

    v_lines = [
        ("Comm State      :", comm_st, st_col),
        ("Channel Config  :", f"Range: {int(v2i_channel.v2i_range) if v2i_channel else 400}px | Lat: {int((v2i_channel.latency if v2i_channel else 0.2)*1000)}ms", (220, 225, 235)),
        ("Packet Stats    :", f"TX: {v2i_channel.packets_sent if v2i_channel else 0} | RX: {v2i_channel.packets_delivered if v2i_channel else 0} | Drop: {v2i_channel.packets_dropped if v2i_channel else 0}", (200, 210, 230)),
        ("RSU Reception   :", "RECEIVED & BUFFERED" if req_received else "AWAITING IN-RANGE TX", (60, 220, 120) if req_received else C_TEXT_DIM),
    ]

    for i, (label, val, col) in enumerate(v_lines):
        ly = v_py + 36 + i * 20
        screen.blit(font_normal.render(label, True, C_TEXT_DIM), (v_px + 14, ly))
        screen.blit(font_bold.render(val, True, col), (v_px + 155, ly))

    if latest_req:
        info_str = f"AMB-01 | ETA: {latest_req.get('eta', 0.0):.1f}s | Priority: {latest_req.get('priority', 'HIGH')}"
        info_col = (255, 235, 100)
    else:
        info_str = "No emergency payload buffered"
        info_col = C_TEXT_DIM

    screen.blit(font_normal.render("Payload Buffer  :", True, C_TEXT_DIM), (v_px + 14, v_py + 118))
    screen.blit(font_bold.render(info_str, True, info_col), (v_px + 155, v_py + 118))

    pygame.draw.line(screen, C_PANEL_BORDER, (v_px + 10, v_py + 144), (v_px + v_panel_w - 10, v_py + 144), 1)
    phase9_chain_badge = "AMB-01 ──(V2I)──> RSU-01 ──(PREEMPT)──> SIGNAL"
    screen.blit(font_small.render(phase9_chain_badge, True, (80, 215, 255)), (v_px + 14, v_py + 154))
    screen.blit(font_normal.render("End-to-End Infrastructure V2I Pipeline", True, C_TEXT_DIM), (v_px + 14, v_py + 178))

    # 4. Right Top Panel: RSU Traffic Conflict & AMB-01 Telemetry
    r_panel_w, r_panel_h = 380, 280
    rx = w - r_panel_w - 20
    ry = 55

    rp_surf = pygame.Surface((r_panel_w, r_panel_h), pygame.SRCALPHA)
    rp_surf.fill(C_PANEL_BG)
    screen.blit(rp_surf, (rx, ry))
    pygame.draw.rect(screen, C_PANEL_BORDER, (rx, ry, r_panel_w, r_panel_h), 2, border_radius=8)

    sig_title = font_bold.render("CONFLICT ANALYSIS & AMB-01 TELEMETRY", True, C_TITLE)
    screen.blit(sig_title, (rx + 14, ry + 10))
    pygame.draw.line(screen, C_PANEL_BORDER, (rx + 10, ry + 30), (rx + r_panel_w - 10, ry + 30), 1)

    # 4-Approach mini overview bar
    n_sig = signal_controller.get_signal("NORTH").value
    s_sig = signal_controller.get_signal("SOUTH").value
    e_sig = signal_controller.get_signal("EAST").value
    w_sig = signal_controller.get_signal("WEST").value
    app_str = f"N: {n_sig[:3]}  S: {s_sig[:3]}  E: {e_sig[:3]}  W: {w_sig[:3]}"
    screen.blit(font_normal.render("Approach Signals :", True, C_TEXT_DIM), (rx + 14, ry + 36))
    screen.blit(font_bold.render(app_str, True, (255, 215, 60) if "YEL" in app_str else ((60, 220, 120) if "GRN" in app_str else (255, 90, 90))), (rx + 160, ry + 36))

    # Conflict zone status
    analysis = getattr(signal_controller, "latest_analysis", None)
    if analysis:
        in_cnt = len(analysis.vehicles_in_conflict)
        if in_cnt > 0:
            cz_str = f"OCCUPIED ({in_cnt} in zone)"
            cz_col = (255, 75, 75)
        elif analysis.conflict_detected:
            cz_str = f"APPROACHING ({len(analysis.vehicles_approaching_conflict)})"
            cz_col = (255, 200, 50)
        else:
            cz_str = "SAFE (CLEAR 0 in zone)"
            cz_col = (60, 225, 130)

        gate_str = "GATE CLEARED" if analysis.safe_for_future_preemption else "GATE HELD (OCCUPIED)"
        gate_col = (60, 225, 130) if analysis.safe_for_future_preemption else (255, 80, 80)
    else:
        cz_str = "NORMAL CYCLE (CLEAR)"
        cz_col = (60, 225, 130)
        gate_str = "INACTIVE (STANDBY)"
        gate_col = C_TEXT_DIM

    screen.blit(font_normal.render("Conflict Zone    :", True, C_TEXT_DIM), (rx + 14, ry + 56))
    screen.blit(font_bold.render(cz_str, True, cz_col), (rx + 160, ry + 56))
    screen.blit(font_normal.render("Preemption Gate  :", True, C_TEXT_DIM), (rx + 14, ry + 76))
    screen.blit(font_bold.render(gate_str, True, gate_col), (rx + 160, ry + 76))

    # AMB-01 Telemetry Sub-panel
    pygame.draw.line(screen, C_PANEL_BORDER, (rx + 10, ry + 98), (rx + r_panel_w - 10, ry + 98), 1)
    screen.blit(font_bold.render("AMB-01 EMERGENCY TELEMETRY", True, (255, 110, 110)), (rx + 14, ry + 104))

    if traffic_manager and traffic_manager.ambulance:
        amb = traffic_manager.ambulance
        kmh = amb.speed * 0.36
        lane_str = "Lane 2 (Overtake x=628)" if abs(amb.x - 628.0) < 15.0 else "Lane 1 (Primary x=592)"
        
        # Determine exact state string
        if getattr(amb, "cleared", False):
            st_text = "CLEARED INTERSECTION"
            st_col = (60, 225, 120)
        elif getattr(amb, "state", None) == VehicleState.EXITING_INTERSECTION:
            st_text = "EXITING_INTERSECTION"
            st_col = (60, 225, 120)
        elif getattr(amb, "in_intersection", False):
            st_text = "CROSSING_INTERSECTION"
            st_col = (60, 225, 120)
        elif getattr(amb, "is_overtaking", False) or getattr(amb, "state", None) == VehicleState.OVERTAKING:
            st_text = "OVERTAKING IN LANE 2"
            st_col = (255, 195, 45)
        elif getattr(amb, "is_authorized", False):
            st_text = "AUTHORIZED_TO_PROCEED" if amb.speed == 0.0 else "ACCELERATING"
            st_col = (80, 220, 255)
        elif amb.speed == 0.0:
            st_text = "WAITING_AT_RED"
            st_col = (255, 80, 80)
        elif amb.braking:
            st_text = "DECELERATING_FOR_RED"
            st_col = (255, 215, 60)
        else:
            st_text = "APPROACHING"
            st_col = (60, 220, 120)

        # Distances to stop line & exit
        lc = traffic_manager.lane_coords.get("SOUTH", {"stop": 490.0, "exit": 350.0})
        d_stop = max(0.0, amb.front_pos[1] - lc["stop"])
        d_exit = max(0.0, amb.rear_pos[1] - lc["exit"])

        amb_lines = [
            ("Vehicle ID / Type:", "AMB-01 (EMERGENCY)", (255, 235, 100)),
            ("Speed / Heading  :", f"{amb.speed:.0f}px/s ({kmh:.0f} km/h) | S -> N", (230, 235, 245)),
            ("Navigation State :", st_text, st_col),
            ("Corridor Lane    :", lane_str, (200, 215, 235)),
            ("Stop / Exit Gap  :", f"Stop: {d_stop:.0f}px | Exit: {d_exit:.0f}px", (180, 200, 220)),
            ("Emergency Priority:", "ACTIVE (SOUTH GREEN)" if getattr(amb, "is_authorized", False) else "AWAITING PREEMPTION", (60, 225, 120) if getattr(amb, "is_authorized", False) else C_TEXT_DIM),
        ]
        for i, (label, val, col) in enumerate(amb_lines):
            ly = ry + 126 + i * 20
            screen.blit(font_normal.render(label, True, C_TEXT_DIM), (rx + 14, ly))
            screen.blit(font_bold.render(val, True, col), (rx + 160, ly))
    else:
        screen.blit(font_normal.render("AMB-01 Status : STANDBY / INACTIVE", True, C_TEXT_DIM), (rx + 14, ry + 130))

    # 5. Right Bottom Panel: Live Event Timeline (Phase 9)
    draw_event_timeline(
        screen=screen,
        event_logger=event_logger,
        font_bold=font_bold,
        font_normal=font_normal,
        font_small=font_small,
        px=rx,
        py=h - 210 - bottom_margin,
        panel_w=r_panel_w,
        panel_h=210,
    )

    # 6. Bottom Controls Banner
    bottom_surf = pygame.Surface((w - 40, 32), pygame.SRCALPHA)
    bottom_surf.fill((10, 15, 25, 210))
    screen.blit(bottom_surf, (20, h - 42))
    pygame.draw.rect(screen, C_PANEL_BORDER, (20, h - 42, w - 40, 32), 1, border_radius=6)

    controls_text = font_normal.render(
        "V2I SMART INTERSECTION (PHASE 9)  |  [SPACE] Pause/Resume    [R] Reset Simulation    [ESC / Q] Launcher",
        True, (190, 205, 225)
    )
    screen.blit(controls_text, (w // 2 - controls_text.get_width() // 2, h - 34))
