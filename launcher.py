"""V2X Demonstration Launcher — Scenario Selection Menu.

Phase 9: Provides a pygame-based selector so a teacher can choose
between the two independent V2X scenarios without touching the terminal.

Usage:
    python launcher.py

Controls:
    [1] or click   → Start V2V Overtaking Collision-Avoidance
    [2] or click   → Start V2I Intersection Collision-Avoidance
    [ESC] / [Q]    → Quit
"""

import subprocess
import sys
import os

import pygame

# ─────────────────────────────────────────────────────────────────────────────
# Visual constants
# ─────────────────────────────────────────────────────────────────────────────
W, H           = 700, 480
FPS            = 60

C_BG_TOP       = (8,  18,  45)
C_BG_BOT       = (15, 35,  80)
C_PANEL        = (20, 30,  65, 220)
C_BORDER       = (60, 100, 200)
C_TITLE        = (200, 220, 255)
C_SUBTITLE     = (140, 160, 210)
C_WHITE        = (240, 240, 240)
C_YELLOW       = (255, 220,  60)
C_GREEN_HOVER  = (50,  200, 120)
C_BLUE_HOVER   = (60,  160, 255)
C_BTN_V2V      = (30,  80,  40)
C_BTN_V2I      = (20,  50, 100)
C_BTN_BORDER   = (80, 200, 140)
C_BTN_BORDER2  = (60, 140, 255)
C_BADGE_V2V    = (60,  200,  80)
C_BADGE_V2I    = (60,  140, 255)
C_DIM          = (120, 120, 140)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient background
# ─────────────────────────────────────────────────────────────────────────────

def draw_gradient(screen) -> None:
    for y in range(H):
        t = y / H
        r = int(C_BG_TOP[0] + (C_BG_BOT[0] - C_BG_TOP[0]) * t)
        g = int(C_BG_TOP[1] + (C_BG_BOT[1] - C_BG_TOP[1]) * t)
        b = int(C_BG_TOP[2] + (C_BG_BOT[2] - C_BG_TOP[2]) * t)
        pygame.draw.line(screen, (r, g, b), (0, y), (W, y))


# ─────────────────────────────────────────────────────────────────────────────
# Animated grid lines
# ─────────────────────────────────────────────────────────────────────────────

_grid_offset = 0.0

def draw_grid(screen, dt: float) -> None:
    global _grid_offset
    _grid_offset = (_grid_offset + 20 * dt) % 40
    for x in range(0, W + 40, 40):
        ox = int(x - _grid_offset)
        pygame.draw.line(screen, (30, 50, 90), (ox, 0), (ox, H), 1)
    for y in range(0, H, 40):
        pygame.draw.line(screen, (30, 50, 90), (0, y), (W, y), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Button
# ─────────────────────────────────────────────────────────────────────────────

class ScenarioButton:
    def __init__(self, rect, label, badge_text, badge_col,
                 border_col, bg_col, description, key_hint):
        self.rect        = pygame.Rect(rect)
        self.label       = label
        self.badge_text  = badge_text
        self.badge_col   = badge_col
        self.border_col  = border_col
        self.bg_col      = bg_col
        self.description = description
        self.key_hint    = key_hint
        self.hovered     = False
        self._glow       = 0.0

    def update(self, mouse_pos, dt) -> None:
        self.hovered = self.rect.collidepoint(mouse_pos)
        target = 1.0 if self.hovered else 0.0
        self._glow += (target - self._glow) * min(1.0, 8 * dt)

    def draw(self, screen, font_lg, font_md, font_sm) -> None:
        r = self.rect

        # Background panel
        surf = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        surf.fill((*self.bg_col, 220))
        screen.blit(surf, r.topleft)

        # Glow border
        glow_intensity = int(100 + 155 * self._glow)
        border_col = tuple(min(255, int(c * (0.5 + 0.5 * self._glow)))
                           for c in self.border_col)
        pygame.draw.rect(screen, border_col, r, 3, border_radius=10)

        # Badge (V2V / V2I)
        badge_r = pygame.Rect(r.x + 14, r.y + 14, 72, 32)
        pygame.draw.rect(screen, self.badge_col, badge_r, border_radius=6)
        badge_txt = font_md.render(self.badge_text, True, (10, 10, 20))
        screen.blit(badge_txt, (badge_r.x + badge_r.w // 2 - badge_txt.get_width() // 2,
                                badge_r.y + badge_r.h // 2 - badge_txt.get_height() // 2))

        # Key hint [1] / [2]
        hint = font_md.render(self.key_hint, True, C_YELLOW)
        screen.blit(hint, (r.right - hint.get_width() - 14, r.y + 18))

        # Label
        lbl = font_lg.render(self.label, True,
                              C_WHITE if not self.hovered else C_YELLOW)
        screen.blit(lbl, (r.x + 14, r.y + 56))

        # Description lines
        for i, line in enumerate(self.description):
            dl = font_sm.render(line, True, C_DIM if not self.hovered else C_WHITE)
            screen.blit(dl, (r.x + 14, r.y + 100 + i * 22))

    def is_clicked(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Launch helper
# ─────────────────────────────────────────────────────────────────────────────

def launch(script: str) -> None:
    """Close the launcher window and start the chosen scenario."""
    pygame.quit()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe  = sys.executable
    cmd = [python_exe, os.path.join(project_dir, script)]
    subprocess.run(cmd, cwd=project_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Main menu loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("V2X Collision-Avoidance Demonstration")
    clock  = pygame.time.Clock()

    font_title = pygame.font.SysFont("consolas", 30, bold=True)
    font_sub   = pygame.font.SysFont("consolas", 16)
    font_lg    = pygame.font.SysFont("consolas", 19, bold=True)
    font_md    = pygame.font.SysFont("consolas", 17, bold=True)
    font_sm    = pygame.font.SysFont("consolas", 15)

    btn_v2v = ScenarioButton(
        rect=(60, 165, W - 120, 130),
        label="V2V  Overtaking Collision-Avoidance",
        badge_text="V2V",
        badge_col=C_BADGE_V2V,
        border_col=C_BTN_BORDER,
        bg_col=C_BTN_V2V,
        description=[
            "Vehicle-to-Vehicle communication prevents a head-on collision",
            "during an overtaking manoeuvre on a highway.",
            "Ego receives V2V broadcast → aborts overtake → collision avoided.",
        ],
        key_hint="[1]",
    )

    btn_v2i = ScenarioButton(
        rect=(60, 315, W - 120, 130),
        label="V2I  Intersection Collision-Avoidance",
        badge_text="V2I",
        badge_col=C_BADGE_V2I,
        border_col=C_BTN_BORDER2,
        bg_col=C_BTN_V2I,
        description=[
            "Smart Traffic Signal / RSU broadcasts collision-risk warning.",
            "Ego receives V2I message → AI evaluates risk → automatic braking.",
            "Cross-traffic passes safely → signal GREEN → ego resumes.",
        ],
        key_hint="[2]",
    )

    footer = "Press [1] or [2]  ·  ESC / Q to quit"

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        mouse = pygame.mouse.get_pos()
        btn_v2v.update(mouse, dt)
        btn_v2i.update(mouse, dt)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_1:
                    launch("main.py")
                    return
                elif event.key == pygame.K_2:
                    launch("v2i_main.py")
                    return
            elif btn_v2v.is_clicked(event):
                launch("main.py")
                return
            elif btn_v2i.is_clicked(event):
                launch("v2i_main.py")
                return

        # ── Draw ──────────────────────────────────────────────────────────────
        draw_gradient(screen)
        draw_grid(screen, dt)

        # Title block
        title = font_title.render("V2X COLLISION-AVOIDANCE DEMONSTRATION", True, C_TITLE)
        screen.blit(title, (W // 2 - title.get_width() // 2, 30))
        sub = font_sub.render("Select a scenario to run", True, C_SUBTITLE)
        screen.blit(sub, (W // 2 - sub.get_width() // 2, 70))

        # Separator
        pygame.draw.line(screen, C_BORDER, (60, 100), (W - 60, 100), 1)

        # Buttons
        btn_v2v.draw(screen, font_lg, font_md, font_sm)
        btn_v2i.draw(screen, font_lg, font_md, font_sm)

        # Footer
        foot = font_sm.render(footer, True, C_DIM)
        screen.blit(foot, (W // 2 - foot.get_width() // 2, H - 30))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
