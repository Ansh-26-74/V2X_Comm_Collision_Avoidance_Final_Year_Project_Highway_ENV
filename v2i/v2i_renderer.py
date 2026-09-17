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
# Drawing: Phase 13B V2V Inter-Vehicle Communication & Maneuver Link
# ─────────────────────────────────────────────────────────────────────────────

def draw_v2v_inter_link(
    screen: pygame.Surface,
    ambulance=None,
    civilian_vehicles=None,
    font_small=None,
) -> None:
    """Render Phase 13B V2V inter-vehicle communication link, TTC warning halo, and maneuver guide."""
    if ambulance is None or civilian_vehicles is None:
        return

    threat_id = getattr(ambulance, "v2v_detected_vehicle_id", None)
    if not threat_id and getattr(ambulance, "latest_v2v_threat", None):
        threat_id = ambulance.latest_v2v_threat.sender_id

    if not threat_id:
        return

    threat_veh = None
    for v in civilian_vehicles:
        if v.vehicle_id == threat_id:
            threat_veh = v
            break

    if threat_veh is None:
        return

    risk = getattr(ambulance, "v2v_risk_state", "NONE")
    ttc = getattr(ambulance, "v2v_ttc", float("inf"))
    is_evading = getattr(ambulance, "is_v2v_evading", False)
    is_braking = getattr(ambulance, "v2v_emergency_braking", False)

    if risk not in ("WARNING", "CRITICAL") and not is_evading and not is_braking:
        return

    if risk == "CRITICAL" or is_braking:
        link_col = (255, 60, 60)
        halo_col = (255, 50, 50, 85)
        badge_border = (255, 80, 80)
    elif is_evading:
        link_col = (255, 165, 35)
        halo_col = (255, 180, 40, 65)
        badge_border = (255, 180, 40)
    else:
        link_col = (255, 215, 60)
        halo_col = (255, 220, 60, 55)
        badge_border = (255, 215, 60)

    # 1. Pulsing threat halo around civilian vehicle C-01
    halo_surf = pygame.Surface((70, 70), pygame.SRCALPHA)
    pulse_r = 24 + int(4 * math.sin(_beacon_phase * 4.0))
    pygame.draw.circle(halo_surf, halo_col, (35, 35), pulse_r)
    pygame.draw.circle(halo_surf, (*link_col, 180), (35, 35), pulse_r, 2)
    screen.blit(halo_surf, (int(threat_veh.x - 35), int(threat_veh.y - 35)))

    # 2. V2V Wireless Link Beam between AMB-01 and C-01 (animated dashed line)
    x1, y1 = int(ambulance.x), int(ambulance.y - ambulance.length / 2)
    x2, y2 = int(threat_veh.x), int(threat_veh.y + threat_veh.length / 2)

    dash_len = 8
    gap_len = 5
    tot = dash_len + gap_len
    dist_link = math.hypot(x2 - x1, y2 - y1)
    if dist_link > 5.0:
        steps = int(dist_link / tot)
        for s in range(steps):
            frac_s = (s * tot) / dist_link
            frac_e = min(1.0, ((s * tot) + dash_len) / dist_link)
            sx = int(x1 + (x2 - x1) * frac_s)
            sy = int(y1 + (y2 - y1) * frac_s)
            ex = int(x1 + (x2 - x1) * frac_e)
            ey = int(y1 + (y2 - y1) * frac_e)
            pygame.draw.line(screen, link_col, (sx, sy), (ex, ey), 2)

    # 3. Midpoint floating V2V badge
    if font_small:
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        ttc_str = f"TTC: {ttc:.2f}s" if ttc < 99.0 else "TTC: SAFE"
        status_tag = "CRITICAL" if risk == "CRITICAL" else ("EVADING" if is_evading else ("BRAKING" if is_braking else "WARNING"))
        badge_txt = f"V2V: AMB-01 ↔ {threat_id} | {ttc_str} [{status_tag}]"
        txt_surf = font_small.render(badge_txt, True, (255, 255, 255))
        bw, bh = txt_surf.get_width() + 12, txt_surf.get_height() + 6
        bg_surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
        bg_surf.fill((16, 22, 36, 220))
        pygame.draw.rect(bg_surf, badge_border, (0, 0, bw, bh), 1, border_radius=4)
        screen.blit(bg_surf, (mid_x - bw // 2, mid_y - bh // 2))
        screen.blit(txt_surf, (mid_x - bw // 2 + 6, mid_y - bh // 2 + 3))

    # 4. If evasive maneuver is active: draw lateral transition arrow to Lane 2
    if is_evading and font_small:
        target_x = getattr(ambulance, "target_lane_x", 628.0)
        arrow_y = int(ambulance.y - ambulance.length / 2 - 25)
        pygame.draw.line(screen, (255, 175, 45), (int(ambulance.x), int(ambulance.y)), (int(target_x), arrow_y), 2)
        pygame.draw.circle(screen, (255, 215, 60), (int(target_x), arrow_y), 4)
        man_lbl = font_small.render("⇗ V2V EVASIVE MANEUVER -> LANE 2", True, (255, 195, 45))
        screen.blit(man_lbl, (int(target_x + 8), arrow_y - 8))

    # 5. If emergency braking fallback is active: draw braking indicator
    if is_braking and font_small:
        stop_line_y = int(threat_veh.y + threat_veh.length / 2 + 22.0)
        pygame.draw.line(screen, (255, 50, 50), (int(ambulance.x - 22), stop_line_y), (int(ambulance.x + 22), stop_line_y), 3)
        brk_lbl = font_small.render("⛔ V2V EMERGENCY BRAKING (LANE BLOCKED)", True, (255, 80, 80))
        screen.blit(brk_lbl, (int(ambulance.x - brk_lbl.get_width() // 2), stop_line_y + 6))


# ─────────────────────────────────────────────────────────────────────────────
# Drawing: Phase 13D AI Trajectory Prediction & Conflict Visualization
# ─────────────────────────────────────────────────────────────────────────────

def draw_ai_trajectory(
    screen: pygame.Surface,
    ambulance=None,
    civilian_vehicles: list | None = None,
    font_small=None,
    font_timer=None,
    show_ai: bool = True,
    traffic_manager=None,
) -> None:
    """Render Phase 13D AI Trajectory Prediction, 6 waypoints, and forecasted conflict region.
    
    Draws:
      - Smooth dashed future trajectory line from tracked vehicle (C-01)
      - Six discrete waypoint markers corresponding to +0.25s .. +1.50s horizons
      - Dynamic alert coloring (vivid amber when predicted_conflict == True, cyan when safe)
      - Pulsing conflict zone marker and forward safety corridor when conflict forecast is active
      - Does NOT mutate any vehicle or simulation state.
    """
    if not show_ai:
        return

    pred = None
    if ambulance is not None:
        pred = getattr(ambulance, "latest_ai_prediction", None)
    if (pred is None or not getattr(pred, "prediction_available", False)) and traffic_manager is not None:
        pred = getattr(traffic_manager, "latest_ai_prediction", None)

    if pred is None or not getattr(pred, "prediction_available", False):
        return

    waypoints = getattr(pred, "waypoints", [])
    if not waypoints:
        return

    # Find target tracked vehicle (typically C-01)
    target_id = getattr(pred, "sender_id", "C-01")
    target_veh = None
    if civilian_vehicles:
        for v in civilian_vehicles:
            if v.vehicle_id == target_id:
                target_veh = v
                break

    # Determine visual styling based on genuine AI conflict prediction
    is_conflict = getattr(pred, "predicted_conflict", False)
    if is_conflict:
        # Alert vivid amber/orange
        spline_col = (255, 110, 35)
        halo_col = (255, 90, 25, 140)
        node_border = (255, 140, 45)
        tag_bg = (35, 16, 12, 225)
        status_txt = "AI PREDICTED CONFLICT"
        status_col = (255, 125, 45)
    else:
        # Normal calm cyan
        spline_col = (45, 205, 240)
        halo_col = (30, 160, 220, 100)
        node_border = (120, 225, 255)
        tag_bg = (12, 24, 38, 220)
        status_txt = "AI FORECAST: CLEAR"
        status_col = (80, 225, 255)

    # 1. Trajectory line: from C-01 front bumper through the 6 predicted waypoints
    if target_veh is not None:
        # C-01 moving South->North (-Y): front bumper is at y - length/2
        start_pt = (int(target_veh.x), int(target_veh.y - target_veh.length / 2))
    else:
        start_pt = (int(waypoints[0].x), int(waypoints[0].y))

    pts = [start_pt] + [(int(wp.x), int(wp.y)) for wp in waypoints]

    # Draw dashed polyline connecting consecutive points
    dash_len = 6
    gap_len = 4
    seg_step = dash_len + gap_len

    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist > 1.0:
            num_dashes = max(1, int(dist / seg_step))
            for d in range(num_dashes):
                f_start = (d * seg_step) / dist
                f_end = min(1.0, ((d * seg_step) + dash_len) / dist)
                sx = int(p1[0] + dx * f_start)
                sy = int(p1[1] + dy * f_start)
                ex = int(p1[0] + dx * f_end)
                ey = int(p1[1] + dy * f_end)
                pygame.draw.line(screen, spline_col, (sx, sy), (ex, ey), 2)

    # 2. Render small waypoint markers (+0.25s .. +1.50s)
    for idx, wp in enumerate(waypoints):
        wx, wy = int(wp.x), int(wp.y)
        is_last = (idx == len(waypoints) - 1)
        r_node = 5 if is_last else 3

        # Glowing halo around node
        pygame.draw.circle(screen, halo_col, (wx, wy), r_node + 3)
        # Node circle
        pygame.draw.circle(screen, spline_col, (wx, wy), r_node)
        pygame.draw.circle(screen, node_border, (wx, wy), r_node, 1)

        # Compact labels
        if font_small:
            if is_last:
                lbl_str = f"+{wp.horizon_offset_s:.2f}s [AI HORIZON]"
                lbl_surf = font_small.render(lbl_str, True, status_col)
                bw, bh = lbl_surf.get_width() + 8, lbl_surf.get_height() + 4
                bg = pygame.Surface((bw, bh), pygame.SRCALPHA)
                bg.fill(tag_bg)
                pygame.draw.rect(bg, node_border, (0, 0, bw, bh), 1, border_radius=3)
                screen.blit(bg, (wx + 10, wy - bh // 2))
                screen.blit(lbl_surf, (wx + 14, wy - bh // 2 + 2))
            else:
                lbl_str = f"{wp.horizon_offset_s:.2f}"
                lbl_surf = font_small.render(lbl_str, True, (190, 210, 230))
                screen.blit(lbl_surf, (wx + 8, wy - 6))

    # 3. Conflict region visualization if predicted conflict is active
    if is_conflict and font_small:
        conflict_wp = waypoints[min(len(waypoints) - 1, 2)]  # ~0.75s future point
        cx, cy = int(conflict_wp.x), int(conflict_wp.y)

        # Glowing conflict ring
        pulse_r = 14 + int(3 * math.sin(_beacon_phase * 5.0))
        c_surf = pygame.Surface((pulse_r * 2 + 10, pulse_r * 2 + 10), pygame.SRCALPHA)
        pygame.draw.circle(c_surf, (255, 60, 40, 90), (pulse_r + 5, pulse_r + 5), pulse_r)
        pygame.draw.circle(c_surf, (255, 110, 35, 220), (pulse_r + 5, pulse_r + 5), pulse_r, 2)
        # Warning diamond/X marker in center
        pygame.draw.line(c_surf, (255, 240, 220), (pulse_r + 5 - 4, pulse_r + 5 - 4), (pulse_r + 5 + 4, pulse_r + 5 + 4), 2)
        pygame.draw.line(c_surf, (255, 240, 220), (pulse_r + 5 - 4, pulse_r + 5 + 4), (pulse_r + 5 + 4, pulse_r + 5 - 4), 2)
        screen.blit(c_surf, (cx - pulse_r - 5, cy - pulse_r - 5))

        # Floating prediction tag (Strictly prediction-oriented wording per Step 5 Part 3)
        tag_surf = font_small.render("⚠️ AI PREDICTED CONFLICT", True, (255, 140, 50))
        tb_w, tb_h = tag_surf.get_width() + 10, tag_surf.get_height() + 6
        t_bg = pygame.Surface((tb_w, tb_h), pygame.SRCALPHA)
        t_bg.fill((25, 14, 12, 230))
        pygame.draw.rect(t_bg, (255, 95, 35), (0, 0, tb_w, tb_h), 1, border_radius=4)
        screen.blit(t_bg, (cx - tb_w // 2, cy - pulse_r - tb_h - 4))
        screen.blit(tag_surf, (cx - tb_w // 2 + 5, cy - pulse_r - tb_h - 1))

        # Ambulance forward safety corridor highlight
        if ambulance is not None:
            amb_front_y = ambulance.y - ambulance.length / 2
            corridor_len = max(35.0, amb_front_y - cy)
            corridor_w = 32
            cor_surf = pygame.Surface((corridor_w, int(corridor_len)), pygame.SRCALPHA)
            cor_surf.fill((255, 130, 40, 24))
            pygame.draw.rect(cor_surf, (255, 140, 45, 85), (0, 0, corridor_w, int(corridor_len)), 1, border_radius=3)
            screen.blit(cor_surf, (int(ambulance.x - corridor_w // 2), int(cy)))


# ─────────────────────────────────────────────────────────────────────────────
# Drawing: RSU / Smart Infrastructure Station
# ─────────────────────────────────────────────────────────────────────────────

RSU_X = 730
RSU_Y = 285
RSU_POS = (float(RSU_X), float(RSU_Y))


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
# Drawing: HUD & Telemetry Panels (Layout-Contained & Bounded)
# ─────────────────────────────────────────────────────────────────────────────

def fit_text_to_width(font, text: str, max_w: int) -> str:
    """Safely truncate text with ellipsis if its rendered width exceeds max_w in pixels."""
    if not text or font.size(text)[0] <= max_w:
        return text
    truncated = text
    while len(truncated) > 3 and font.size(truncated + "..")[0] > max_w:
        truncated = truncated[:-1]
    return truncated + ".."


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
    traffic_manager=None,
) -> None:
    """Render the Phase 9 chronological event timeline audit log with Phase 13D AI metrics."""
    p_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    p_surf.fill(C_PANEL_BG)
    screen.blit(p_surf, (px, py))
    pygame.draw.rect(screen, C_PANEL_BORDER, (px, py, panel_w, panel_h), 2, border_radius=8)

    # Header with title & lead badge
    lead_time = None
    if traffic_manager and hasattr(traffic_manager, "safety_fusion") and traffic_manager.safety_fusion:
        lead_time = getattr(traffic_manager.safety_fusion, "ai_lead_time", None)

    lead_w = 0
    if lead_time is not None and lead_time > 0.0:
        lead_badge = font_small.render(f"AI LEAD: +{lead_time:.2f}s", True, (80, 240, 140))
        lead_w = lead_badge.get_width()
        screen.blit(lead_badge, (px + panel_w - lead_w - 12, py + 9))

    title_max_w = panel_w - lead_w - 28
    title_text = "V2I/V2V EVENT TIMELINE (AUDIT LOG)"
    if font_bold.size(title_text)[0] > title_max_w:
        title_text = "EVENT TIMELINE (AUDIT)"
    title = font_bold.render(fit_text_to_width(font_bold, title_text, title_max_w), True, (255, 195, 60))
    screen.blit(title, (px + 12, py + 8))

    pygame.draw.line(screen, C_PANEL_BORDER, (px + 8, py + 27), (px + panel_w - 8, py + 27), 1)

    events = event_logger.get_events() if event_logger else []
    if not events:
        screen.blit(font_normal.render("Awaiting system state transitions...", True, C_TEXT_DIM), (px + 12, py + 38))
        return

    # Dynamic row allocation based on available height inside panel
    title_h = 30
    bottom_pad = 6
    avail_h = panel_h - title_h - bottom_pad
    row_h = 18
    max_events = max(1, avail_h // row_h)
    visible_events = events[-max_events:]

    # Clip boundary to strictly protect panel borders
    prev_clip = screen.get_clip()
    screen.set_clip(pygame.Rect(px + 2, py + 28, panel_w - 4, panel_h - 30))

    msg_max_w = panel_w - 106
    for i, evt in enumerate(visible_events):
        ey = py + 31 + i * row_h

        # Category tag and color mapping
        if "AI" in evt.event_type or evt.category == EventCategory.SYSTEM:
            cat_col = (180, 140, 255)
            badge = "AI  "
        elif evt.category == EventCategory.COMMUNICATION:
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
        screen.blit(font_small.render(ts_str, True, (160, 175, 195)), (px + 10, ey))

        # Category badge
        badge_surf = font_small.render(f"[{badge}]", True, cat_col)
        screen.blit(badge_surf, (px + 52, ey))

        # Message (strictly truncated to never escape card)
        msg_str = fit_text_to_width(font_small, evt.message, msg_max_w)
        screen.blit(font_small.render(msg_str, True, (230, 235, 245)), (px + 98, ey))

    screen.set_clip(prev_clip)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing: Phase 13D AI & Safety Fusion HUD Cards
# ─────────────────────────────────────────────────────────────────────────────

def draw_ai_hud_card(
    screen: pygame.Surface,
    traffic_manager,
    font_bold,
    font_normal,
    font_small,
    px: int = 16,
    py: int = 54,
    panel_w: int = 370,
    panel_h: int = 132,
) -> None:
    """Render Phase 13D dedicated AI Trajectory Prediction HUD Card."""
    surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    surf.fill(C_PANEL_BG)
    screen.blit(surf, (px, py))
    pygame.draw.rect(screen, (70, 110, 180), (px, py, panel_w, panel_h), 2, border_radius=8)

    amb = getattr(traffic_manager, "ambulance", None) if traffic_manager else None
    pred = getattr(amb, "latest_ai_prediction", None) if amb else None
    if pred is None and traffic_manager:
        pred = getattr(traffic_manager, "latest_ai_prediction", None)
    predictor = getattr(traffic_manager, "ai_predictor", None) if traffic_manager else None

    # Tracked vehicle ID
    v2v_threat_id = getattr(amb, "v2v_detected_vehicle_id", None) if amb else None
    if not v2v_threat_id and amb and getattr(amb, "latest_v2v_threat", None):
        v2v_threat_id = amb.latest_v2v_threat.sender_id
    tracked_veh = getattr(pred, "sender_id", None) or v2v_threat_id or "C-01"

    # History sample count
    history_count = 0
    if predictor and hasattr(predictor, "history_buffers"):
        history_count = len(predictor.history_buffers.get(tracked_veh, []))

    pred_available = getattr(pred, "prediction_available", False) if pred else False
    is_conflict = getattr(pred, "predicted_conflict", False) if pred else False
    ai_loading = getattr(traffic_manager, "ai_loading", False) if traffic_manager else False

    if predictor is None and ai_loading:
        status_txt = "INITIALIZING"
        status_badge = "INIT"
        status_col = (255, 215, 60)
        pred_txt = "INITIALIZING (PyTorch)"
        pred_col = (255, 215, 60)
        risk_txt = "INITIALIZING (PyTorch MLP)"
        risk_col = (255, 215, 60)
        clr_str = "N/A"
        inf_str = "N/A"
    elif predictor is None:
        err_msg = getattr(traffic_manager, "ai_error", None)
        status_txt = "UNAVAILABLE"
        status_badge = "UNAVAIL"
        status_col = (180, 190, 205)
        pred_txt = f"UNAVAILABLE ({err_msg[:12]})" if err_msg else "UNAVAILABLE"
        pred_col = (180, 190, 205)
        risk_txt = "UNAVAILABLE (DETERMINISTIC V2V)"
        risk_col = (255, 215, 60)
        clr_str = "N/A"
        inf_str = "N/A"
    elif not pred_available:
        status_txt = f"COLLECTING DATA ({history_count}/5)"
        status_badge = f"{history_count}/5 PKTS"
        status_col = (255, 215, 60)
        pred_txt = f"AWAITING TELEMETRY ({history_count}/5)"
        pred_col = (255, 215, 60)
        risk_txt = "AWAITING TELEMETRY (3-5 PKTS)"
        risk_col = C_TEXT_DIM
        clr_str = "N/A"
        inf_str = "N/A"
    elif is_conflict:
        status_txt = "ACTIVE"
        status_badge = "ACTIVE"
        status_col = (60, 225, 130)
        pred_txt = "PREDICTED CONFLICT"
        pred_col = (255, 110, 35)
        risk_txt = "PREDICTED CONFLICT"
        risk_col = (255, 110, 35)
        clr_str = f"{pred.predicted_min_distance:.1f} px"
        inf_str = f"{pred.inference_time_ms:.2f} ms"
    else:
        status_txt = "ACTIVE"
        status_badge = "ACTIVE"
        status_col = (60, 225, 130)
        pred_txt = "PREDICTED CLEAR / SAFE"
        pred_col = (60, 225, 130)
        risk_txt = "PREDICTED SAFE"
        risk_col = (60, 225, 130)
        clr_str = f"{pred.predicted_min_distance:.1f} px" if pred.predicted_min_distance < 999.0 else "CLEAR"
        inf_str = f"{pred.inference_time_ms:.2f} ms"

    # Header
    title_surf = font_bold.render("AI TRAJECTORY PREDICTION", True, (90, 205, 255))
    screen.blit(title_surf, (px + 12, py + 7))
    badge_surf = font_small.render(f"[{status_badge}]", True, status_col)
    screen.blit(badge_surf, (px + panel_w - badge_surf.get_width() - 12, py + 8))
    pygame.draw.line(screen, (55, 85, 140), (px + 8, py + 26), (px + panel_w - 8, py + 26), 1)

    # Content lines safely bound inside card
    val_x = px + 126
    val_max_w = panel_w - 138

    lines = [
        ("Tracked Vehicle :", f"{tracked_veh} (V2V Telemetry)", (230, 235, 245), False),
        ("Prediction State:", pred_txt, pred_col, True),
        ("Forecast Horizon:", "1.50s (6 waypoints @ 10 Hz)", (200, 215, 235), False),
        ("AI Risk State   :", risk_txt, risk_col, True),
        ("Min Clearance   :", clr_str, (255, 215, 60) if is_conflict else (200, 215, 235), False),
        ("Inference Lat.  :", f"{inf_str} | PyTorch MLP", (160, 230, 200) if pred_available else C_TEXT_DIM, False),
    ]

    prev_clip = screen.get_clip()
    screen.set_clip(pygame.Rect(px + 2, py + 27, panel_w - 4, panel_h - 29))

    avail_h = panel_h - 32 - 4
    row_h = min(16, avail_h // 6)
    for i, (lbl, val, col, is_bold) in enumerate(lines):
        ly = py + 29 + i * row_h
        screen.blit(font_small.render(lbl, True, C_TEXT_DIM), (px + 12, ly))
        val_fitted = fit_text_to_width(font_bold if is_bold else font_small, val, val_max_w)
        screen.blit(font_bold.render(val_fitted, True, col) if is_bold else font_small.render(val_fitted, True, col), (val_x, ly))

    screen.set_clip(prev_clip)


def draw_safety_fusion_hud_card(
    screen: pygame.Surface,
    traffic_manager,
    font_bold,
    font_normal,
    font_small,
    px: int = 16,
    py: int = 190,
    panel_w: int = 370,
    panel_h: int = 152,
) -> None:
    """Render Phase 13D Safety Fusion & AI/TTC Distinction HUD Card."""
    surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    surf.fill(C_PANEL_BG)
    screen.blit(surf, (px, py))

    amb = getattr(traffic_manager, "ambulance", None) if traffic_manager else None
    fusion = getattr(amb, "latest_safety_fusion", None) if amb else None
    fusion_engine = getattr(traffic_manager, "safety_fusion", None) if traffic_manager else None

    # Determine border and highlight color from fused state
    fused_state = getattr(fusion, "fused_risk_state", "SAFE") if fusion else "SAFE"
    if fused_state == "CRITICAL":
        border_col = (255, 75, 75)
        badge_col = (255, 65, 65)
        badge_text = "CRITICAL"
    elif fused_state == "ELEVATED_WARNING":
        border_col = (255, 160, 40)
        badge_col = (255, 175, 45)
        badge_text = "ELEVATED"
    elif fused_state == "WARNING":
        border_col = (255, 215, 60)
        badge_col = (255, 215, 60)
        badge_text = "WARNING"
    elif fused_state == "ADVISORY_MONITORING":
        border_col = (80, 180, 240)
        badge_col = (80, 200, 255)
        badge_text = "ADVISORY"
    else:
        border_col = (60, 140, 95)
        badge_col = (60, 225, 130)
        badge_text = "SAFE"

    pygame.draw.rect(screen, border_col, (px, py, panel_w, panel_h), 2, border_radius=8)

    # Header
    title_surf = font_bold.render("SAFETY FUSION (ARBITRATION)", True, (240, 200, 110))
    screen.blit(title_surf, (px + 12, py + 6))
    badge_surf = font_small.render(f"[{badge_text}]", True, badge_col)
    screen.blit(badge_surf, (px + panel_w - badge_surf.get_width() - 12, py + 7))
    pygame.draw.line(screen, (55, 85, 140), (px + 8, py + 24), (px + panel_w - 8, py + 24), 1)

    # Content extraction
    ai_risk = getattr(fusion, "ai_risk_state", "SAFE") if fusion else "STANDBY"
    ttc_val = getattr(fusion, "ttc", float("inf")) if fusion else float("inf")
    ttc_risk = getattr(fusion, "ttc_risk_state", "NONE") if fusion else "NONE"
    clr_val = getattr(fusion, "deterministic_clearance", 0.0) if fusion else 0.0
    lane2_ok = getattr(fusion, "target_lane_safe", True) if fusion else True
    action = getattr(fusion, "recommended_action", "SAFE_CRUISE") if fusion else "SAFE_CRUISE"
    raw_reason = getattr(fusion, "decision_reason", "BASELINE_SAFE") if fusion else "INITIALIZING"

    # Compact decision reason mapping
    reason_map = {
        "AI_CONFLICT_WITH_TTC_WARNING": "AI CONFLICT + TTC WARN",
        "BLOCKED_LANE_OVERRIDE": "BLOCKED LANE OVERRIDE",
        "CONFLICT_ZONE_OVERRIDE": "CONFLICT ZONE OVERRIDE",
        "TTC_CRITICAL_OVERRIDE": "TTC CRITICAL OVERRIDE",
        "BASELINE_SAFE": "BASELINE SAFE",
        "AI_CONFLICT_MONITORING": "AI CONFLICT MONITOR",
        "TTC_CRITICAL": "TTC CRITICAL",
    }
    reason = reason_map.get(raw_reason, raw_reason.replace("_", " "))

    # Lead time
    lead_time = getattr(fusion_engine, "ai_lead_time", None) if fusion_engine else None
    if lead_time is not None and lead_time > 0.0:
        lead_str = f"+{lead_time:.2f}s before TTC"
        lead_col = (80, 240, 140)
    else:
        lead_str = "N/A"
        lead_col = C_TEXT_DIM

    ttc_str = f"{ttc_val:.2f}s ({ttc_risk})" if ttc_val < 99.0 else "SAFE (>4.0s)"
    lane_str = "SAFE" if lane2_ok else "BLOCKED"

    val_x = px + 126
    val_max_w = panel_w - 138

    lines = [
        ("AI Risk Forecast:", ai_risk.replace("_", " "), (255, 110, 35) if "CONFLICT" in ai_risk else (60, 225, 130), False),
        ("Kinematic TTC    :", ttc_str, (255, 75, 75) if ttc_risk == "CRITICAL" else ((255, 215, 60) if ttc_risk == "WARNING" else (60, 225, 130)), False),
        ("Clearance / L2   :", f"{clr_val:.1f} px | Lane 2: {lane_str}", (230, 235, 245), False),
        ("FUSED RISK STATE :", fused_state, badge_col, True),
        ("RECOMMENDED ACT  :", action.replace("_", " "), (255, 195, 45) if "EVASIVE" in action else (60, 225, 130), True),
        ("DECISION REASON  :", reason, (200, 215, 235), False),
        ("AI Early Warning :", lead_str, lead_col, False),
    ]

    prev_clip = screen.get_clip()
    screen.set_clip(pygame.Rect(px + 2, py + 25, panel_w - 4, panel_h - 43))

    avail_h = panel_h - 26 - 19
    row_h = min(15, avail_h // 7)
    for i, (lbl, val, col, is_bold) in enumerate(lines):
        ly = py + 26 + i * row_h
        screen.blit(font_small.render(lbl, True, C_TEXT_DIM), (px + 12, ly))
        val_fitted = fit_text_to_width(font_bold if is_bold else font_small, val, val_max_w)
        screen.blit(font_bold.render(val_fitted, True, col) if is_bold else font_small.render(val_fitted, True, col), (val_x, ly))

    screen.set_clip(prev_clip)

    # Explanatory footer banner safely positioned
    foot_y = py + panel_h - 17
    pygame.draw.line(screen, (45, 65, 95), (px + 8, foot_y - 2), (px + panel_w - 8, foot_y - 2), 1)
    foot_txt = "AI: ADVISORY FORECAST | TTC: AUTHORITATIVE SAFETY"
    foot_fitted = fit_text_to_width(font_small, foot_txt, panel_w - 16)
    foot_surf = font_small.render(foot_fitted, True, (120, 185, 235))
    screen.blit(foot_surf, (px + (panel_w - foot_surf.get_width()) // 2, foot_y))


def draw_rsu_combined_panel(
    screen: pygame.Surface,
    signal_controller,
    v2i_channel,
    traffic_manager,
    font_bold,
    font_normal,
    font_small,
    v_px: int = 16,
    v_py: int = 458,
    v_panel_w: int = 370,
    v_panel_h: int = 224,
) -> None:
    """Render combined Bottom-Left RSU Infrastructure and V2X Communication panel."""
    v_surf = pygame.Surface((v_panel_w, v_panel_h), pygame.SRCALPHA)
    v_surf.fill(C_PANEL_BG)
    screen.blit(v_surf, (v_px, v_py))
    pygame.draw.rect(screen, C_PANEL_BORDER, (v_px, v_py, v_panel_w, v_panel_h), 2, border_radius=8)

    v2i_title = font_bold.render("SMART INTERSECTION & V2X LINK", True, (80, 215, 255))
    screen.blit(v2i_title, (v_px + 12, v_py + 6))
    pygame.draw.line(screen, C_PANEL_BORDER, (v_px + 8, v_py + 25), (v_px + v_panel_w - 8, v_py + 25), 1)

    # Signal status
    curr_phase = signal_controller.current_phase
    phase_name = curr_phase.value.replace("_", " ")
    time_left = signal_controller.time_remaining
    total_dur = signal_controller.phase_duration

    if curr_phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
        preempt_badge = "ACTIVE (SOUTH GREEN)"
        preempt_col = (60, 220, 120)
    elif curr_phase == CyclePhase.EMERGENCY_TERMINATING:
        preempt_badge = "TERMINATING (YELLOW)"
        preempt_col = (255, 215, 60)
    elif curr_phase in (CyclePhase.RECOVERY_ALL_RED, CyclePhase.PREEMPTION_ALL_RED):
        preempt_badge = "ALL-RED CLEARANCE"
        preempt_col = (255, 80, 80)
    else:
        preempt_badge = "NORMAL TIMED CYCLE"
        preempt_col = C_TEXT_DIM

    comm_st = v2i_channel.comm_state if v2i_channel else "OFFLINE"
    st_col = (60, 220, 120) if comm_st in ("DELIVERED", "IN_RANGE", "TRANSMITTED") else (
        (255, 215, 60) if comm_st == "OUT_OF_RANGE" else (255, 80, 80)
    )

    req_received = signal_controller.emergency_request_received
    latest_req = signal_controller.latest_emergency_request
    if latest_req:
        info_str = f"ETA: {latest_req.get('eta', 0.0):.1f}s | Pri: {latest_req.get('priority', 'HIGH')}"
    else:
        info_str = "No payload buffered"

    amb = traffic_manager.ambulance if traffic_manager else None
    v2v_threat_id = getattr(amb, "v2v_detected_vehicle_id", None) if amb else None
    if not v2v_threat_id and amb and getattr(amb, "latest_v2v_threat", None):
        v2v_threat_id = amb.latest_v2v_threat.sender_id
    v2v_peer = v2v_threat_id or "C-01"

    val_x = v_px + 126
    val_max_w = v_panel_w - 138

    lines = [
        ("Intersection ID :", f"{signal_controller.intersection_id} | Preempt: {preempt_badge}", preempt_col, True),
        ("Signal Phase    :", f"{phase_name} ({time_left:04.1f}s/{total_dur:04.1f}s)", (240, 240, 240), False),
        ("Safety Mutex    :", "✓ MUTEX ENFORCED (NO CONFLICT GREEN)", (60, 225, 130), True),
        ("V2I Comm State  :", f"{comm_st} | RSU Range: 400px", st_col, False),
        ("RSU Packets     :", f"TX: {v2i_channel.packets_sent if v2i_channel else 0} | RX: {v2i_channel.packets_delivered if v2i_channel else 0} | Drop: {v2i_channel.packets_dropped if v2i_channel else 0}", (200, 210, 230), False),
        ("RSU Reception   :", "RECEIVED & BUFFERED" if req_received else "AWAITING IN-RANGE TX", (60, 220, 120) if req_received else C_TEXT_DIM, False),
        ("V2I Payload     :", info_str, (255, 235, 100) if latest_req else C_TEXT_DIM, False),
        ("V2V Peer Link   :", f"AMB-01 ↔ {v2v_peer} (10 Hz DSRC / 200px)", (120, 210, 255), False),
    ]

    prev_clip = screen.get_clip()
    screen.set_clip(pygame.Rect(v_px + 2, v_py + 26, v_panel_w - 4, v_panel_h - 66))

    row_h = 17
    for i, (lbl, val, col, is_bold) in enumerate(lines):
        ly = v_py + 28 + i * row_h
        screen.blit(font_small.render(lbl, True, C_TEXT_DIM), (v_px + 12, ly))
        val_fitted = fit_text_to_width(font_bold if is_bold else font_small, val, val_max_w)
        screen.blit(font_bold.render(val_fitted, True, col) if is_bold else font_small.render(val_fitted, True, col), (val_x, ly))

    screen.set_clip(prev_clip)

    # Footer banner
    foot_div_y = v_py + v_panel_h - 40
    pygame.draw.line(screen, C_PANEL_BORDER, (v_px + 8, foot_div_y), (v_px + v_panel_w - 8, foot_div_y), 1)
    phase9_chain_badge = "V2I: RSU Preemption  |  V2V: Collision Avoidance"
    b_fitted = fit_text_to_width(font_small, phase9_chain_badge, v_panel_w - 24)
    screen.blit(font_small.render(b_fitted, True, (80, 215, 255)), (v_px + 12, foot_div_y + 4))
    sub_fitted = fit_text_to_width(font_normal, "Integrated Dual-Layer Connected Vehicle System", v_panel_w - 24)
    screen.blit(font_normal.render(sub_fitted, True, C_TEXT_DIM), (v_px + 12, foot_div_y + 18))


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
    show_ai: bool = True,
) -> None:
    """Draw the smart intersection controller HUD and telemetry panels."""
    w, h = screen.get_size()
    if font_small is None:
        font_small = pygame.font.SysFont("consolas", 12, bold=True)

    # Standard layout margins and dimensions
    margin_x = 16
    panel_w = 370
    rx = w - panel_w - margin_x
    px = margin_x

    # 1. Top Title Banner
    banner_h = 38
    title_surf = pygame.Surface((w - 2 * margin_x, banner_h), pygame.SRCALPHA)
    title_surf.fill(C_PANEL_BG)
    screen.blit(title_surf, (margin_x, 8))
    pygame.draw.rect(screen, C_PANEL_BORDER, (margin_x, 8, w - 2 * margin_x, banner_h), 1, border_radius=6)

    title_text = "INTEGRATED V2X SMART INTERSECTION — V2I PREEMPTION & V2V AVOIDANCE (PHASE 13)"
    title_max_w = w - 2 * margin_x - 240
    title_fitted = fit_text_to_width(font_title, title_text, title_max_w)
    title_render = font_title.render(title_fitted, True, C_TITLE)
    screen.blit(title_render, (margin_x + 14, 16))

    pause_badge = " [PAUSED]" if paused else " [RUNNING]"
    time_text = font_bold.render(f"SIM TIME: {sim_time:05.1f}s {pause_badge}", True, (255, 215, 60) if paused else (60, 220, 120))
    screen.blit(time_text, (w - margin_x - time_text.get_width() - 14, 17))

    # Controls bar at bottom
    controls_h = 28
    controls_y = h - controls_h - 6
    bottom_gap = 6
    bot_panel_h = 224
    bot_panel_y = controls_y - bottom_gap - bot_panel_h

    # 2 & 3. Left Side Panels (Dual AI & Safety Fusion Cards vs Classic RSU)
    if show_ai:
        # Upper Left: Dedicated AI Trajectory Prediction HUD Card
        draw_ai_hud_card(
            screen=screen,
            traffic_manager=traffic_manager,
            font_bold=font_bold,
            font_normal=font_normal,
            font_small=font_small,
            px=px,
            py=52,
            panel_w=panel_w,
            panel_h=132,
        )

        # Mid Left: Dedicated Safety Fusion & AI/TTC Distinction Card
        draw_safety_fusion_hud_card(
            screen=screen,
            traffic_manager=traffic_manager,
            font_bold=font_bold,
            font_normal=font_normal,
            font_small=font_small,
            px=px,
            py=188,
            panel_w=panel_w,
            panel_h=152,
        )

        # Bottom Left: Combined RSU Infrastructure & V2X Link
        draw_rsu_combined_panel(
            screen=screen,
            signal_controller=signal_controller,
            v2i_channel=v2i_channel,
            traffic_manager=traffic_manager,
            font_bold=font_bold,
            font_normal=font_normal,
            font_small=font_small,
            v_px=px,
            v_py=bot_panel_y,
            v_panel_w=panel_w,
            v_panel_h=bot_panel_h,
        )
    else:
        # Classic RSU Controller Panel (when AI toggle is OFF)
        c_panel_h = 250
        p_surf = pygame.Surface((panel_w, c_panel_h), pygame.SRCALPHA)
        p_surf.fill(C_PANEL_BG)
        screen.blit(p_surf, (px, 52))
        pygame.draw.rect(screen, C_PANEL_BORDER, (px, 52, panel_w, c_panel_h), 2, border_radius=8)

        rsu_title = font_bold.render("SMART INTERSECTION / RSU-01", True, C_TITLE)
        screen.blit(rsu_title, (px + 12, 52 + 8))
        pygame.draw.line(screen, C_PANEL_BORDER, (px + 8, 52 + 28), (px + panel_w - 8, 52 + 28), 1)

        curr_phase = signal_controller.current_phase
        phase_name = curr_phase.value.replace("_", " ")
        time_left = signal_controller.time_remaining
        total_dur = signal_controller.phase_duration

        if curr_phase == CyclePhase.EMERGENCY_SOUTH_GREEN:
            preempt_badge = "ACTIVE (SOUTH GREEN)"
            preempt_col = (60, 220, 120)
            mode_str = "Mode: EMERGENCY PREEMPTION (SOUTH PRIORITY)"
        elif curr_phase == CyclePhase.EMERGENCY_TERMINATING:
            preempt_badge = "TERMINATING (SOUTH YELLOW)"
            preempt_col = (255, 215, 60)
            mode_str = "Mode: EMERGENCY TERMINATING (YELLOW CLEARANCE)"
        elif curr_phase in (CyclePhase.RECOVERY_ALL_RED, CyclePhase.PREEMPTION_ALL_RED):
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

        val_x = px + 140
        val_max_w = panel_w - 152
        for i, (label, val, col) in enumerate(lines):
            ly = 52 + 34 + i * 19
            screen.blit(font_normal.render(label, True, C_TEXT_DIM), (px + 12, ly))
            val_fitted = fit_text_to_width(font_bold, val, val_max_w)
            screen.blit(font_bold.render(val_fitted, True, col), (val_x, ly))

        progress = 1.0 - (time_left / max(total_dur, 0.1))
        bar_x, bar_y, bar_w, bar_h = px + 12, 52 + 154, panel_w - 24, 10
        pygame.draw.rect(screen, (30, 40, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        bar_fill_col = (255, 210, 40) if "YELLOW" in phase_name else (60, 220, 120)
        pygame.draw.rect(screen, bar_fill_col, (bar_x, bar_y, int(bar_w * progress), bar_h), border_radius=5)
        pygame.draw.rect(screen, C_PANEL_BORDER, (bar_x, bar_y, bar_w, bar_h), 1, border_radius=5)

        safe_badge = "✓ MUTEX ENFORCED (NO CONFLICTING GREEN)"
        safe_surf = font_normal.render(fit_text_to_width(font_normal, safe_badge, panel_w - 24), True, (60, 225, 130))
        screen.blit(safe_surf, (px + 12, 52 + 176))

        mode_txt = font_normal.render(fit_text_to_width(font_normal, mode_str, panel_w - 24), True, (255, 215, 60) if getattr(signal_controller, "preemption_active", False) or getattr(signal_controller, "preemption_clearing", False) or getattr(signal_controller, "emergency_terminating", False) else C_TEXT_DIM)
        screen.blit(mode_txt, (px + 12, 52 + 200))

        # Classic Bottom-Left Panel: V2I Wireless Communication & Telemetry
        v_surf = pygame.Surface((panel_w, bot_panel_h), pygame.SRCALPHA)
        v_surf.fill(C_PANEL_BG)
        screen.blit(v_surf, (px, bot_panel_y))
        pygame.draw.rect(screen, C_PANEL_BORDER, (px, bot_panel_y, panel_w, bot_panel_h), 2, border_radius=8)

        v2i_title = font_bold.render("V2X WIRELESS (V2I RSU + V2V PEER)", True, (80, 215, 255))
        screen.blit(v2i_title, (px + 12, bot_panel_y + 6))
        pygame.draw.line(screen, C_PANEL_BORDER, (px + 8, bot_panel_y + 25), (px + panel_w - 8, bot_panel_y + 25), 1)

        comm_st = v2i_channel.comm_state if v2i_channel else "OFFLINE"
        st_col = (60, 220, 120) if comm_st in ("DELIVERED", "IN_RANGE", "TRANSMITTED") else (
            (255, 215, 60) if comm_st == "OUT_OF_RANGE" else (255, 80, 80)
        )

        req_received = signal_controller.emergency_request_received
        latest_req = signal_controller.latest_emergency_request

        v_lines = [
            ("V2I Comm State  :", comm_st, st_col),
            ("RSU Link (400px):", f"TX: {v2i_channel.packets_sent if v2i_channel else 0} | RX: {v2i_channel.packets_delivered if v2i_channel else 0} | Drop: {v2i_channel.packets_dropped if v2i_channel else 0}", (200, 210, 230)),
            ("RSU Reception   :", "RECEIVED & BUFFERED" if req_received else "AWAITING IN-RANGE TX", (60, 220, 120) if req_received else C_TEXT_DIM),
        ]

        for i, (label, val, col) in enumerate(v_lines):
            ly = bot_panel_y + 30 + i * 18
            screen.blit(font_normal.render(label, True, C_TEXT_DIM), (px + 12, ly))
            val_fitted = fit_text_to_width(font_bold, val, val_max_w)
            screen.blit(font_bold.render(val_fitted, True, col), (val_x, ly))

        if latest_req:
            info_str = f"ETA: {latest_req.get('eta', 0.0):.1f}s | Priority: {latest_req.get('priority', 'HIGH')}"
            info_col = (255, 235, 100)
        else:
            info_str = "No payload buffered"
            info_col = C_TEXT_DIM

        screen.blit(font_normal.render("V2I Payload     :", True, C_TEXT_DIM), (px + 12, bot_panel_y + 86))
        info_fitted = fit_text_to_width(font_bold, info_str, val_max_w)
        screen.blit(font_bold.render(info_fitted, True, info_col), (val_x, bot_panel_y + 86))

        pygame.draw.line(screen, C_PANEL_BORDER, (px + 8, bot_panel_y + 108), (px + panel_w - 8, bot_panel_y + 108), 1)

        amb = traffic_manager.ambulance if traffic_manager else None
        v2v_threat_id = getattr(amb, "v2v_detected_vehicle_id", None) if amb else None
        if not v2v_threat_id and amb and getattr(amb, "latest_v2v_threat", None):
            v2v_threat_id = amb.latest_v2v_threat.sender_id
        v2v_peer = v2v_threat_id or "C-01"
        v2v_risk = getattr(amb, "v2v_risk_state", "NONE") if amb else "NONE"
        v2v_ttc = getattr(amb, "v2v_ttc", float("inf")) if amb else float("inf")
        is_evading = getattr(amb, "is_v2v_evading", False) if amb else False
        is_braking = getattr(amb, "v2v_emergency_braking", False) if amb else False
        evasion_done = getattr(amb, "v2v_evasion_complete", False) if amb else False

        if v2v_risk == "CRITICAL" or is_braking:
            v2v_badge_col = (255, 65, 65)
            v2v_status_txt = f"CRITICAL (TTC: {v2v_ttc:.2f}s)" if v2v_ttc < 99.0 else "CRITICAL RISK"
        elif is_evading:
            v2v_badge_col = (255, 175, 45)
            v2v_status_txt = f"EVASIVE MANEUVER (TTC: {v2v_ttc:.2f}s)"
        elif evasion_done:
            v2v_badge_col = (60, 225, 130)
            v2v_status_txt = "EVASION COMPLETE (SAFE)"
        elif v2v_risk == "WARNING":
            v2v_badge_col = (255, 215, 60)
            v2v_status_txt = f"WARNING (TTC: {v2v_ttc:.2f}s)"
        elif v2v_risk == "SAFE":
            v2v_badge_col = (60, 225, 130)
            v2v_status_txt = f"SAFE (TTC: {v2v_ttc:.1f}s)"
        else:
            v2v_badge_col = C_TEXT_DIM
            v2v_status_txt = "STANDBY / MONITORING"

        v2v_title_surf = font_bold.render(f"V2V PEER LINK : AMB-01 ↔ {v2v_peer}", True, (255, 185, 45) if v2v_risk in ("WARNING", "CRITICAL") or is_evading or is_braking else (120, 210, 255))
        screen.blit(v2v_title_surf, (px + 12, bot_panel_y + 114))

        screen.blit(font_normal.render("V2V Threat/Risk :", True, C_TEXT_DIM), (px + 12, bot_panel_y + 134))
        threat_fitted = fit_text_to_width(font_bold, v2v_status_txt, val_max_w)
        screen.blit(font_bold.render(threat_fitted, True, v2v_badge_col), (val_x, bot_panel_y + 134))

        if is_evading:
            resp_txt = "⇗ LATERAL SHIFT TO LANE 2"
            resp_col = (255, 195, 45)
        elif is_braking:
            resp_txt = "⛔ EMERGENCY BRAKING (BLOCKED)"
            resp_col = (255, 75, 75)
        elif evasion_done:
            resp_txt = "✓ CORRIDOR CLEAR (CRUISING)"
            resp_col = (60, 225, 130)
        elif v2v_risk == "CRITICAL":
            resp_txt = "⚡ EVASIVE MANEUVER TRIGGERED"
            resp_col = (255, 75, 75)
        else:
            resp_txt = "SAFE FOLLOWING / CRUISE"
            resp_col = C_TEXT_DIM

        screen.blit(font_normal.render("V2V Response    :", True, C_TEXT_DIM), (px + 12, bot_panel_y + 154))
        resp_fitted = fit_text_to_width(font_bold, resp_txt, val_max_w)
        screen.blit(font_bold.render(resp_fitted, True, resp_col), (val_x, bot_panel_y + 154))

        pygame.draw.line(screen, C_PANEL_BORDER, (px + 8, bot_panel_y + 176), (px + panel_w - 8, bot_panel_y + 176), 1)
        phase9_chain_badge = "V2I: RSU Preemption  |  V2V: Collision Avoidance"
        screen.blit(font_small.render(phase9_chain_badge, True, (80, 215, 255)), (px + 12, bot_panel_y + 184))
        screen.blit(font_normal.render("Integrated Dual-Layer Connected Vehicle System", True, C_TEXT_DIM), (px + 12, bot_panel_y + 204))

    # 4. Right Top Panel: RSU Traffic Conflict & AMB-01 Telemetry
    r_panel_h = 265
    ry = 52

    rp_surf = pygame.Surface((panel_w, r_panel_h), pygame.SRCALPHA)
    rp_surf.fill(C_PANEL_BG)
    screen.blit(rp_surf, (rx, ry))
    pygame.draw.rect(screen, C_PANEL_BORDER, (rx, ry, panel_w, r_panel_h), 2, border_radius=8)

    sig_title = font_bold.render("CONFLICT ANALYSIS & AMB-01 TELEMETRY", True, C_TITLE)
    screen.blit(sig_title, (rx + 12, ry + 8))
    pygame.draw.line(screen, C_PANEL_BORDER, (rx + 8, ry + 27), (rx + panel_w - 8, ry + 27), 1)

    r_val_x = rx + 138
    r_val_max_w = panel_w - 150

    prev_clip = screen.get_clip()
    screen.set_clip(pygame.Rect(rx + 2, ry + 28, panel_w - 4, r_panel_h - 30))

    # 4-Approach mini overview bar
    n_sig = signal_controller.get_signal("NORTH").value
    s_sig = signal_controller.get_signal("SOUTH").value
    e_sig = signal_controller.get_signal("EAST").value
    w_sig = signal_controller.get_signal("WEST").value
    app_str = f"N: {n_sig[:3]}  S: {s_sig[:3]}  E: {e_sig[:3]}  W: {w_sig[:3]}"
    screen.blit(font_normal.render("Approach Signals :", True, C_TEXT_DIM), (rx + 12, ry + 32))
    screen.blit(font_bold.render(app_str, True, (255, 215, 60) if "YEL" in app_str else ((60, 220, 120) if "GRN" in app_str else (255, 90, 90))), (r_val_x, ry + 32))

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

    screen.blit(font_normal.render("Conflict Zone    :", True, C_TEXT_DIM), (rx + 12, ry + 51))
    cz_fitted = fit_text_to_width(font_bold, cz_str, r_val_max_w)
    screen.blit(font_bold.render(cz_fitted, True, cz_col), (r_val_x, ry + 51))

    screen.blit(font_normal.render("Preemption Gate  :", True, C_TEXT_DIM), (rx + 12, ry + 70))
    gate_fitted = fit_text_to_width(font_bold, gate_str, r_val_max_w)
    screen.blit(font_bold.render(gate_fitted, True, gate_col), (r_val_x, ry + 70))

    # AMB-01 Telemetry Sub-panel
    pygame.draw.line(screen, C_PANEL_BORDER, (rx + 8, ry + 90), (rx + panel_w - 8, ry + 90), 1)
    screen.blit(font_bold.render("AMB-01 EMERGENCY TELEMETRY", True, (255, 110, 110)), (rx + 12, ry + 95))

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
        elif getattr(amb, "is_v2v_evading", False) or getattr(amb, "state", None) == VehicleState.V2V_EVASIVE_MANEUVER:
            st_text = "V2V EVASIVE MANEUVER"
            st_col = (255, 165, 35)
        elif getattr(amb, "v2v_emergency_braking", False) or getattr(amb, "state", None) == VehicleState.V2V_EMERGENCY_BRAKING:
            st_text = "V2V EMERGENCY BRAKING"
            st_col = (255, 60, 60)
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

        v2v_risk_val = getattr(amb, "v2v_risk_state", "NONE")
        v2v_ttc_val = getattr(amb, "v2v_ttc", float("inf"))
        if v2v_risk_val == "CRITICAL":
            v2v_txt = f"CRITICAL ({v2v_ttc_val:.2f}s)"
            v2v_col = (255, 65, 65)
        elif v2v_risk_val == "WARNING":
            v2v_txt = f"WARNING ({v2v_ttc_val:.2f}s)"
            v2v_col = (255, 215, 60)
        elif getattr(amb, "v2v_evasion_complete", False):
            v2v_txt = "EVASION COMPLETE (SAFE)"
            v2v_col = (60, 225, 130)
        elif v2v_risk_val == "SAFE":
            v2v_txt = f"SAFE ({v2v_ttc_val:.1f}s)"
            v2v_col = (60, 225, 130)
        else:
            v2v_txt = "STANDBY / CLEAR"
            v2v_col = C_TEXT_DIM

        amb_lines = [
            ("Vehicle ID / Type:", "AMB-01 (EMERGENCY)", (255, 235, 100)),
            ("Speed / Heading  :", f"{amb.speed:.0f}px/s ({kmh:.0f} km/h) | S -> N", (230, 235, 245)),
            ("Navigation State :", st_text, st_col),
            ("Corridor Lane    :", lane_str, (200, 215, 235)),
            ("V2V Threat / TTC :", v2v_txt, v2v_col),
            ("Emergency Priority:", "ACTIVE (SOUTH GREEN)" if getattr(amb, "is_authorized", False) else "AWAITING PREEMPTION", (60, 225, 120) if getattr(amb, "is_authorized", False) else C_TEXT_DIM),
        ]
        for i, (label, val, col) in enumerate(amb_lines):
            ly = ry + 115 + i * 19
            screen.blit(font_normal.render(label, True, C_TEXT_DIM), (rx + 12, ly))
            val_fitted = fit_text_to_width(font_bold, val, r_val_max_w)
            screen.blit(font_bold.render(val_fitted, True, col), (r_val_x, ly))
    else:
        screen.blit(font_normal.render("AMB-01 Status : STANDBY / INACTIVE", True, C_TEXT_DIM), (rx + 12, ry + 120))

    screen.set_clip(prev_clip)

    # 5. Right Bottom Panel: Live Event Timeline
    draw_event_timeline(
        screen=screen,
        event_logger=event_logger,
        font_bold=font_bold,
        font_normal=font_normal,
        font_small=font_small,
        px=rx,
        py=bot_panel_y,
        panel_w=panel_w,
        panel_h=bot_panel_h,
        traffic_manager=traffic_manager,
    )

    # 6. Bottom Controls Banner
    bottom_surf = pygame.Surface((w - 2 * margin_x, controls_h), pygame.SRCALPHA)
    bottom_surf.fill((10, 15, 25, 210))
    screen.blit(bottom_surf, (margin_x, controls_y))
    pygame.draw.rect(screen, C_PANEL_BORDER, (margin_x, controls_y, w - 2 * margin_x, controls_h), 1, border_radius=6)

    controls_text = "INTEGRATED V2X SMART INTERSECTION (PHASE 13)  |  [SPACE] Pause/Resume    [R] Reset    [I] Toggle AI    [ESC / Q] Launcher"
    controls_fitted = fit_text_to_width(font_normal, controls_text, w - 2 * margin_x - 16)
    controls_render = font_normal.render(controls_fitted, True, (190, 205, 225))
    screen.blit(controls_render, (w // 2 - controls_render.get_width() // 2, controls_y + 5))

