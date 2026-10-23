"""
Visual Frontend Engine using Pygame.
Phase 1: Advanced UI & Data Generation Engine.

Features:
- Bounded 2D arena with grid coordinate markings, glowing boundaries, and obstacles.
- Rich robot visualization: directional vectors, heading arrows, motion trails,
  faint target destinations, direct path lines, and dynamic risk auras.
- Floating pairwise threat indicators with real-time TTC labels.
- Glassmorphic side panel displaying live telemetry, coordinates, velocities,
  compass headings, dataset recording status, and risk distribution graphs.
"""

import math
import pygame
from typing import List, Tuple, Dict, Any, Optional

from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    ARENA_X,
    ARENA_Y,
    ARENA_WIDTH,
    ARENA_HEIGHT,
    PANEL_X,
    PANEL_Y,
    PANEL_WIDTH,
    PANEL_HEIGHT,
    COLOR_BG_DARK,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_PANEL_CARD,
    COLOR_ARENA_BG,
    COLOR_ARENA_BORDER,
    COLOR_ARENA_GRID,
    COLOR_ARENA_ACCENT,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_ACCENT,
    COLOR_SAFE,
    COLOR_WARNING,
    COLOR_HIGH_RISK,
    COLOR_OBSTACLE_FILL,
    COLOR_OBSTACLE_BORDER,
    COLOR_OBSTACLE_HAZARD,
    RISK_LABEL_SAFE,
    RISK_LABEL_WARNING,
    RISK_LABEL_HIGH_RISK,
    RISK_NAMES,
)
from robot_state import Robot, StaticObstacle


class UIEngine:
    """
    High-fidelity rendering engine for multi-agent simulation and telemetry.
    """

    def __init__(self, screen: Optional[pygame.Surface] = None):
        if screen is None:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Intelligent Collision-Risk Prediction Engine | Multi-Robot Navigation")
        else:
            self.screen = screen

        # Initialize Fonts
        pygame.font.init()
        # Prefer modern sans-serif fonts; fall back to standard fonts
        sys_fonts = pygame.font.get_fonts()
        chosen_font = None
        for candidate in ["menlo", "consolas", "sfnsmono", "dejavusansmono", "couriernew", "arial"]:
            if candidate in sys_fonts:
                chosen_font = candidate
                break

        self.font_title = pygame.font.SysFont(chosen_font, 18, bold=True)
        self.font_subtitle = pygame.font.SysFont(chosen_font, 13)
        self.font_body = pygame.font.SysFont(chosen_font, 12)
        self.font_body_bold = pygame.font.SysFont(chosen_font, 12, bold=True)
        self.font_small = pygame.font.SysFont(chosen_font, 10)
        self.font_hud = pygame.font.SysFont(chosen_font, 11)

        # Pulse timer for animations
        self.anim_time: float = 0.0

    def render(
        self,
        sim_time: float,
        fps: float,
        scenario_name: str,
        robots: List[Robot],
        obstacles: List[StaticObstacle],
        pairwise_summaries: List[Dict[str, Any]],
        stats: Dict[str, Any],
        is_paused: bool = False,
        deflection_enabled: bool = False,
        total_collisions: int = 0,
        total_avoidances: int = 0,
        gnn_result: Optional[Any] = None,
    ) -> None:
        """
        Master render cycle: Clears window, draws arena, robots, telemetry, and overlay.
        """
        self.anim_time = sim_time
        self.screen.fill(COLOR_BG_DARK)

        # 1. Draw Arena Bounded Room & Environmental Grid
        self._draw_arena_background()
        self._draw_static_obstacles(obstacles)

        # 2. Draw Target Destinations and Planned Path Lines
        self._draw_targets_and_paths(robots)

        # 3. Draw Robot Motion Trails
        self._draw_motion_trails(robots)

        # 4. Draw Pairwise Conflict Lines (Warning & High Risk)
        self._draw_threat_connections(robots, pairwise_summaries)

        # 5. Draw Autonomous Robots with Heading Vectors, Steering Vectors & Evasive Highlighting
        self._draw_robots(robots, deflection_enabled=deflection_enabled)

        # 6. Draw Telemetry Side Panel with Controller Status & Reliable Metrics
        self._draw_side_panel(
            sim_time=sim_time,
            fps=fps,
            scenario_name=scenario_name,
            robots=robots,
            pairwise_summaries=pairwise_summaries,
            stats=stats,
            is_paused=is_paused,
            deflection_enabled=deflection_enabled,
            total_collisions=total_collisions,
            total_avoidances=total_avoidances,
            gnn_result=gnn_result,
        )

        # 7. Draw Active Avoidance Banner if genuine evasion is underway
        any_evading = any(getattr(r, "is_deflecting", False) for r in robots) and deflection_enabled
        if any_evading:
            self._draw_avoidance_banner()

        pygame.display.flip()

    def _draw_avoidance_banner(self) -> None:
        """Displays prominent avoidance status banner strictly while evasion is genuinely active."""
        banner_text = "COLLISION RISK — AVOIDANCE ACTIVE"
        b_surf = self.font_subtitle.render(banner_text, True, (0, 240, 255))
        bw, bh = b_surf.get_width(), b_surf.get_height()
        bx = ARENA_X + (ARENA_WIDTH - bw) // 2
        by = ARENA_Y + 12
        b_rect = pygame.Rect(bx - 12, by - 4, bw + 24, bh + 8)
        pygame.draw.rect(self.screen, (10, 25, 40), b_rect)
        pygame.draw.rect(self.screen, (0, 240, 255), b_rect, 1)
        self.screen.blit(b_surf, (bx, by))

    def _draw_arena_background(self) -> None:
        """Draws the bounded simulation arena with high-tech floor grid and borders."""
        # Arena Floor
        arena_rect = pygame.Rect(ARENA_X, ARENA_Y, ARENA_WIDTH, ARENA_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_ARENA_BG, arena_rect)

        # Subtle Floor Grid Lines
        grid_step = 40
        for x in range(ARENA_X, ARENA_X + ARENA_WIDTH, grid_step):
            is_accent = (x - ARENA_X) % 160 == 0
            color = COLOR_ARENA_ACCENT if is_accent else COLOR_ARENA_GRID
            pygame.draw.line(self.screen, color, (x, ARENA_Y), (x, ARENA_Y + ARENA_HEIGHT), 1)

        for y in range(ARENA_Y, ARENA_Y + ARENA_HEIGHT, grid_step):
            is_accent = (y - ARENA_Y) % 160 == 0
            color = COLOR_ARENA_ACCENT if is_accent else COLOR_ARENA_GRID
            pygame.draw.line(self.screen, color, (ARENA_X, y), (ARENA_X + ARENA_WIDTH, y), 1)

        # Center Crosshair Markers
        cx = ARENA_X + ARENA_WIDTH // 2
        cy = ARENA_Y + ARENA_HEIGHT // 2
        pygame.draw.line(self.screen, (35, 50, 75), (cx - 20, cy), (cx + 20, cy), 2)
        pygame.draw.line(self.screen, (35, 50, 75), (cx, cy - 20), (cx, cy + 20), 2)

        # Glowing Arena Boundary Border
        pygame.draw.rect(self.screen, COLOR_ARENA_BORDER, arena_rect, 2)

        # Corner Tech Brackets
        corner_len = 16
        corners = [
            (ARENA_X, ARENA_Y),
            (ARENA_X + ARENA_WIDTH, ARENA_Y),
            (ARENA_X, ARENA_Y + ARENA_HEIGHT),
            (ARENA_X + ARENA_WIDTH, ARENA_Y + ARENA_HEIGHT),
        ]
        bracket_color = (0, 240, 255)
        # Top-Left
        pygame.draw.line(self.screen, bracket_color, (ARENA_X, ARENA_Y), (ARENA_X + corner_len, ARENA_Y), 3)
        pygame.draw.line(self.screen, bracket_color, (ARENA_X, ARENA_Y), (ARENA_X, ARENA_Y + corner_len), 3)
        # Top-Right
        pygame.draw.line(self.screen, bracket_color, (ARENA_X + ARENA_WIDTH - corner_len, ARENA_Y), (ARENA_X + ARENA_WIDTH, ARENA_Y), 3)
        pygame.draw.line(self.screen, bracket_color, (ARENA_X + ARENA_WIDTH, ARENA_Y), (ARENA_X + ARENA_WIDTH, ARENA_Y + corner_len), 3)
        # Bottom-Left
        pygame.draw.line(self.screen, bracket_color, (ARENA_X, ARENA_Y + ARENA_HEIGHT), (ARENA_X + corner_len, ARENA_Y + ARENA_HEIGHT), 3)
        pygame.draw.line(self.screen, bracket_color, (ARENA_X, ARENA_Y + ARENA_HEIGHT), (ARENA_X, ARENA_Y + ARENA_HEIGHT - corner_len), 3)
        # Bottom-Right
        pygame.draw.line(self.screen, bracket_color, (ARENA_X + ARENA_WIDTH - corner_len, ARENA_Y + ARENA_HEIGHT), (ARENA_X + ARENA_WIDTH, ARENA_Y + ARENA_HEIGHT), 3)
        pygame.draw.line(self.screen, bracket_color, (ARENA_X + ARENA_WIDTH, ARENA_Y + ARENA_HEIGHT), (ARENA_X + ARENA_WIDTH, ARENA_Y + ARENA_HEIGHT - corner_len), 3)

        # Arena Dimension Label
        dim_surf = self.font_small.render(f"BOUNDED ZONE: {ARENA_WIDTH} x {ARENA_HEIGHT} px", True, (60, 80, 110))
        self.screen.blit(dim_surf, (ARENA_X + 10, ARENA_Y + 10))

    def _draw_static_obstacles(self, obstacles: List[StaticObstacle]) -> None:
        """Renders static pillar obstacles with industrial styling."""
        for obs in obstacles:
            ox = int(obs.x)
            oy = int(obs.y)
            r = int(obs.radius)

            # Outer protective shadow ring
            pygame.draw.circle(self.screen, (20, 26, 38), (ox, oy), r + 4)
            # Main pillar body
            pygame.draw.circle(self.screen, COLOR_OBSTACLE_FILL, (ox, oy), r)
            # Border outline
            pygame.draw.circle(self.screen, COLOR_OBSTACLE_BORDER, (ox, oy), r, 2)
            # Inner core
            pygame.draw.circle(self.screen, (45, 55, 72), (ox, oy), max(4, r - 12), 1)

            # Hazard Cross / Center Accent
            pygame.draw.line(self.screen, COLOR_OBSTACLE_HAZARD, (ox - 8, oy), (ox + 8, oy), 2)
            pygame.draw.line(self.screen, COLOR_OBSTACLE_HAZARD, (ox, oy - 8), (ox, oy + 8), 2)

            # Obstacle Label
            lbl = self.font_small.render(f"OBS #{obs.id}", True, (140, 155, 175))
            lbl_rect = lbl.get_rect(center=(ox, oy + r + 12))
            self.screen.blit(lbl, lbl_rect)

    def _draw_targets_and_paths(self, robots: List[Robot]) -> None:
        """Draws faint target destinations, original routes, and visible active deflected trajectories."""
        for r in robots:
            rx, ry = int(r.x), int(r.y)
            tx, ty = int(r.target_x), int(r.target_y)

            is_evading = getattr(r, "is_deflecting", False)
            deflected_pts = getattr(r, "deflected_path", [])

            if is_evading and deflected_pts:
                # 1A. Original Intended Route (Dimmed to visually distinguish from deflected trajectory)
                self._draw_dashed_line(
                    self.screen,
                    (110, 125, 145),
                    (rx, ry),
                    (tx, ty),
                    dash_len=6,
                    space_len=8,
                    alpha=40,
                )
                orig_lbl = self.font_small.render("ORIGINAL ROUTE", True, (110, 125, 145))
                mx = (rx + tx) // 2
                my = (ry + ty) // 2
                self.screen.blit(orig_lbl, (mx - orig_lbl.get_width() // 2, my - 8))

                # 1B. Actual Deflected Trajectory (Smooth glowing curved arc)
                pts = [(rx, ry)] + [(int(p[0]), int(p[1])) for p in deflected_pts]
                if len(pts) >= 2:
                    curve_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                    # Outer soft glow
                    for i in range(len(pts) - 1):
                        pygame.draw.line(curve_surf, (0, 240, 255, 60), pts[i], pts[i + 1], 6)
                        pygame.draw.line(curve_surf, (0, 240, 255, 230), pts[i], pts[i + 1], 2)
                        pygame.draw.circle(curve_surf, (0, 240, 255, 200), pts[i + 1], 3)

                    self.screen.blit(curve_surf, (0, 0))

                    # Connect final projected waypoint back to target destination
                    last_pt = pts[-1]
                    self._draw_dashed_line(
                        self.screen,
                        (0, 240, 255),
                        last_pt,
                        (tx, ty),
                        dash_len=5,
                        space_len=5,
                        alpha=120,
                    )

                    # Pill label near curve apex
                    apex_pt = pts[min(3, len(pts) - 1)]
                    badge_surf = self.font_small.render("DEFLECTED PATH", True, (0, 240, 255))
                    bw, bh = badge_surf.get_width(), badge_surf.get_height()
                    b_rect = pygame.Rect(apex_pt[0] - bw // 2 - 4, apex_pt[1] - bh // 2 - 2, bw + 8, bh + 4)
                    pygame.draw.rect(self.screen, (10, 20, 32), b_rect)
                    pygame.draw.rect(self.screen, (0, 240, 255), b_rect, 1)
                    self.screen.blit(badge_surf, (b_rect.x + 4, b_rect.y + 2))

            else:
                # 1. Normal Planned Direct Path Line (Dashed appearance)
                self._draw_dashed_line(
                    self.screen, r.color, (rx, ry), (tx, ty), dash_len=8, space_len=6, alpha=80
                )

            # 2. Target Destination Marker
            # Pulsing radius
            pulse = math.sin(self.anim_time * 4.0 + r.id) * 3.0
            outer_r = int(max(6, 10 + pulse))

            # Faint target halo
            target_surf = pygame.Surface((outer_r * 2 + 4, outer_r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(target_surf, (*r.color, 45), (outer_r + 2, outer_r + 2), outer_r)
            pygame.draw.circle(target_surf, (*r.color, 160), (outer_r + 2, outer_r + 2), outer_r, 1)
            pygame.draw.circle(target_surf, (*r.color, 240), (outer_r + 2, outer_r + 2), 3)
            # Crosshairs
            pygame.draw.line(target_surf, (*r.color, 160), (outer_r + 2 - outer_r, outer_r + 2), (outer_r + 2 + outer_r, outer_r + 2), 1)
            pygame.draw.line(target_surf, (*r.color, 160), (outer_r + 2, outer_r + 2 - outer_r), (outer_r + 2, outer_r + 2 + outer_r), 1)

            self.screen.blit(target_surf, (tx - outer_r - 2, ty - outer_r - 2))

            # Target ID Badge
            t_lbl = self.font_small.render(f"Goal {r.id}", True, r.color)
            self.screen.blit(t_lbl, (tx + 12, ty - 6))

    def _draw_dashed_line(
        self,
        surf: pygame.Surface,
        color: Tuple[int, int, int],
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        dash_len: int = 8,
        space_len: int = 6,
        alpha: int = 90,
    ) -> None:
        """Draws an alpha-blended dashed line between two points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            return

        ux = dx / dist
        uy = dy / dist
        step = dash_len + space_len
        num_steps = int(dist / step)

        line_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        curr = 0.0
        for _ in range(num_steps):
            sx = p1[0] + ux * curr
            sy = p1[1] + uy * curr
            ex = p1[0] + ux * min(curr + dash_len, dist)
            ey = p1[1] + uy * min(curr + dash_len, dist)
            pygame.draw.line(line_surf, (*color, alpha), (sx, sy), (ex, ey), 1)
            curr += step

        surf.blit(line_surf, (0, 0))

    def _draw_motion_trails(self, robots: List[Robot]) -> None:
        """Renders fading historical motion trails showing each robot's past path."""
        for r in robots:
            pts = list(r.trail)
            if len(pts) < 2:
                continue

            # Draw consecutive line segments with increasing alpha
            trail_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            n = len(pts)
            for i in range(n - 1):
                alpha = int((i / n) * 95)  # Fade from 0 to 95
                p_start = (int(pts[i][0]), int(pts[i][1]))
                p_end = (int(pts[i + 1][0]), int(pts[i + 1][1]))
                pygame.draw.line(trail_surf, (*r.color, alpha), p_start, p_end, 2)

            self.screen.blit(trail_surf, (0, 0))

    def _draw_threat_connections(self, robots: List[Robot], summaries: List[Dict[str, Any]]) -> None:
        """
        Draws dynamic warning / danger lines between interacting robot pairs
        when Time-To-Collision (TTC) falls into Warning or High Risk tiers.
        """
        robot_dict = {r.id: r for r in robots}
        active_threats = [s for s in summaries if s["risk_label"] != RISK_LABEL_SAFE and not math.isinf(s["ttc"])]
        # Sort by urgency (lowest TTC first)
        active_threats.sort(key=lambda s: s["ttc"])

        threat_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        rendered_badge_centers = []

        for item in active_threats:
            ttc = item["ttc"]
            risk_lbl = item["risk_label"]

            r_i = robot_dict.get(item["pair"][0])
            r_j = robot_dict.get(item["pair"][1])
            if not r_i or not r_j:
                continue

            color = COLOR_HIGH_RISK if risk_lbl == RISK_LABEL_HIGH_RISK else COLOR_WARNING
            width = 3 if risk_lbl == RISK_LABEL_HIGH_RISK else 1

            # Pulsing alpha
            pulse = (math.sin(self.anim_time * 12.0) + 1.0) * 0.5
            alpha = int(140 + pulse * 115) if risk_lbl == RISK_LABEL_HIGH_RISK else 120

            pygame.draw.line(threat_surf, (*color, alpha), (int(r_i.x), int(r_i.y)), (int(r_j.x), int(r_j.y)), width)

        self.screen.blit(threat_surf, (0, 0))

        # Render floating TTC Badges for top most critical threats
        for item in active_threats[:3]:
            ttc = item["ttc"]
            risk_lbl = item["risk_label"]
            r_i = robot_dict.get(item["pair"][0])
            r_j = robot_dict.get(item["pair"][1])
            if not r_i or not r_j:
                continue

            # Don't draw text badge if robots are too close together to avoid cluttering chassis
            if item.get("dist", 999.0) < 65.0:
                continue

            color = COLOR_HIGH_RISK if risk_lbl == RISK_LABEL_HIGH_RISK else COLOR_WARNING
            mx = int((r_i.x + r_j.x) / 2)
            my = int((r_i.y + r_j.y) / 2)

            # Avoid direct overlap with existing badges
            for bx, by in rendered_badge_centers:
                if math.hypot(mx - bx, my - by) < 24:
                    my += 18

            rendered_badge_centers.append((mx, my))

            ttc_text = f"TTC: {ttc:.2f}s"
            badge_surf = self.font_small.render(ttc_text, True, (255, 255, 255))
            bw, bh = badge_surf.get_size()

            bg_rect = pygame.Rect(mx - bw // 2 - 4, my - bh // 2 - 2, bw + 8, bh + 4)
            pygame.draw.rect(self.screen, (15, 20, 30), bg_rect)
            pygame.draw.rect(self.screen, color, bg_rect, 1)
            self.screen.blit(badge_surf, (mx - bw // 2, my - bh // 2))

    def _draw_robots(self, robots: List[Robot], deflection_enabled: bool = False) -> None:
        """
        Renders rich robot chassis, directional heading indicator (triangle/arrow),
        velocity vector, active evasive steering vectors, and risk/evasion halos.
        """
        for r in robots:
            rx = int(r.x)
            ry = int(r.y)
            radius = int(r.radius)

            # 1. Dynamic Risk Halo (Aura)
            halo_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            halo_cx, halo_cy = radius * 2, radius * 2

            if r.risk_label == RISK_LABEL_HIGH_RISK:
                # Urgent High-Risk Pulsing Crimson Halo
                pulse = (math.sin(self.anim_time * 14.0 + r.id) + 1.0) * 0.5
                halo_r = int(radius * 1.5 + pulse * 7)
                pygame.draw.circle(halo_surf, (*COLOR_HIGH_RISK, 80), (halo_cx, halo_cy), halo_r)
                pygame.draw.circle(halo_surf, (*COLOR_HIGH_RISK, 220), (halo_cx, halo_cy), halo_r, 2)
            elif r.risk_label == RISK_LABEL_WARNING:
                # Warning Pulsing Amber Halo
                pulse = (math.sin(self.anim_time * 8.0 + r.id) + 1.0) * 0.5
                halo_r = int(radius * 1.3 + pulse * 4)
                pygame.draw.circle(halo_surf, (*COLOR_WARNING, 50), (halo_cx, halo_cy), halo_r)
                pygame.draw.circle(halo_surf, (*COLOR_WARNING, 180), (halo_cx, halo_cy), halo_r, 1)
            else:
                # Subtle Safe Cyan/Green Halo
                halo_r = int(radius * 1.2)
                pygame.draw.circle(halo_surf, (*COLOR_SAFE, 25), (halo_cx, halo_cy), halo_r)

            self.screen.blit(halo_surf, (rx - halo_cx, ry - halo_cy))

            # 1b. Evasive Robot Highlighting (Phase 4 Stage D)
            is_evading = getattr(r, "is_deflecting", False) and deflection_enabled
            if is_evading:
                ev_pulse = (math.sin(self.anim_time * 16.0 + r.id) + 1.0) * 0.5
                ev_r = int(radius * 1.65 + ev_pulse * 5)
                # Outer evasive cyber-cyan aura ring
                pygame.draw.circle(self.screen, (0, 240, 255), (rx, ry), ev_r, 2)
                # Mini "AVOIDANCE ACTIVE" pill above robot
                ev_surf = self.font_small.render("AVOIDANCE ACTIVE", True, (0, 240, 255))
                self.screen.blit(ev_surf, (rx - ev_surf.get_width() // 2, ry + radius + 3))

            # 2. Robot Chassis Body
            # Outer dark ring
            pygame.draw.circle(self.screen, (20, 27, 40), (rx, ry), radius)
            # Main Robot Accent Color Rim
            pygame.draw.circle(self.screen, r.color, (rx, ry), radius, 2)
            # Inner dark core
            pygame.draw.circle(self.screen, (10, 15, 24), (rx, ry), radius - 4)

            # 3. Clear Directional Heading Indicator (Triangle / Arrow Pointer)
            theta = r.theta
            tip_dist = radius + 6
            side_dist = radius - 4
            tip_x = rx + math.cos(theta) * tip_dist
            tip_y = ry + math.sin(theta) * tip_dist

            side1_x = rx + math.cos(theta + 2.5) * side_dist
            side1_y = ry + math.sin(theta + 2.5) * side_dist
            side2_x = rx + math.cos(theta - 2.5) * side_dist
            side2_y = ry + math.sin(theta - 2.5) * side_dist

            # Draw directional triangle pointer on top of chassis
            pygame.draw.polygon(self.screen, r.color, [(tip_x, tip_y), (side1_x, side1_y), (side2_x, side2_y)])

            # 4. Heading Velocity Vector Line extending forward (White)
            vec_length = max(18.0, min(50.0, (r.speed / r.max_speed) * 45.0 + 15.0))
            vx_end = rx + math.cos(theta) * vec_length
            vy_end = ry + math.sin(theta) * vec_length
            pygame.draw.line(self.screen, (255, 255, 255), (rx, ry), (vx_end, vy_end), 2)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(vx_end), int(vy_end)), 2)

            # 4b. Active Deflection Steering Vector Line (Phase 4 Stage D)
            blend = getattr(r, "deflection_blend", 0.0)
            if deflection_enabled and (is_evading or blend > 0.05):
                steer_deg = getattr(r, "deflection_angle_deg", math.degrees(theta))
                steer_rad = math.radians(steer_deg)
                steer_len = max(24.0, min(55.0, (r.speed / r.max_speed) * 48.0 + 18.0))
                sx_end = rx + math.cos(steer_rad) * steer_len
                sy_end = ry + math.sin(steer_rad) * steer_len
                # Vibrant Cyan steering vector
                steer_col = (0, 240, 255)
                pygame.draw.line(self.screen, steer_col, (rx, ry), (sx_end, sy_end), 2)
                # Arrowhead on steering vector
                arr_len = 7
                pygame.draw.line(self.screen, steer_col, (sx_end, sy_end),
                                 (sx_end - math.cos(steer_rad - 0.45) * arr_len,
                                  sy_end - math.sin(steer_rad - 0.45) * arr_len), 2)
                pygame.draw.line(self.screen, steer_col, (sx_end, sy_end),
                                 (sx_end - math.cos(steer_rad + 0.45) * arr_len,
                                  sy_end - math.sin(steer_rad + 0.45) * arr_len), 2)

            # 5. Robot ID Text Label
            id_surf = self.font_body_bold.render(str(r.id), True, (255, 255, 255))
            id_rect = id_surf.get_rect(center=(rx, ry))
            self.screen.blit(id_surf, id_rect)

            # 6. Directional Heading Badge (North, South, etc.)
            heading_str = f"{r.compass_direction} {int(r.heading_deg)}°"
            head_surf = self.font_small.render(heading_str, True, COLOR_TEXT_MUTED)
            self.screen.blit(head_surf, (rx - head_surf.get_width() // 2, ry - radius - 14))

    def _draw_side_panel(
        self,
        sim_time: float,
        fps: float,
        scenario_name: str,
        robots: List[Robot],
        pairwise_summaries: List[Dict[str, Any]],
        stats: Dict[str, Any],
        is_paused: bool,
        deflection_enabled: bool = False,
        total_collisions: int = 0,
        total_avoidances: int = 0,
        gnn_result: Optional[Any] = None,
    ) -> None:
        """
        Renders real-time telemetry side panel: active robots, coordinates,
        velocities, headings, recording statistics, and conflict risk states.
        """
        panel_rect = pygame.Rect(PANEL_X, PANEL_Y, PANEL_WIDTH, PANEL_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, panel_rect, 1)

        curr_y = PANEL_Y + 12

        # Header Title
        title_surf = self.font_title.render("COLLISION-RISK ENGINE", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title_surf, (PANEL_X + 14, curr_y))
        curr_y += 20

        sub_surf = self.font_subtitle.render("Phase 4: GNN Prediction & Adaptive Deflection", True, COLOR_TEXT_ACCENT)
        self.screen.blit(sub_surf, (PANEL_X + 14, curr_y))
        curr_y += 24

        # Scenario Badge
        scen_badge = f"SCENARIO: {scenario_name}"
        scen_surf = self.font_small.render(scen_badge, True, (209, 213, 219))
        scen_bg = pygame.Rect(PANEL_X + 14, curr_y, PANEL_WIDTH - 28, 20)
        pygame.draw.rect(self.screen, (31, 41, 55), scen_bg)
        pygame.draw.rect(self.screen, (55, 65, 81), scen_bg, 1)
        self.screen.blit(scen_surf, (PANEL_X + 22, curr_y + 3))
        curr_y += 24

        # GNN Adaptive Deflection Status & Real-time Collision / Avoidance Counter (Phase 4 Stage D)
        any_evading = any(getattr(r, "is_deflecting", False) for r in robots) and deflection_enabled
        if any_evading:
            deflect_text = "COLLISION RISK — AVOIDANCE ACTIVE"
            deflect_color = (0, 240, 255)
            deflect_bg = (10, 36, 50)
            deflect_border = (0, 240, 255)
        elif deflection_enabled:
            deflect_text = "AI DEFLECTION: [ACTIVE (ON)]"
            deflect_color = COLOR_SAFE
            deflect_bg = (16, 40, 32)
            deflect_border = (16, 185, 129)
        else:
            deflect_text = "AI DEFLECTION: [DISABLED (OFF)]"
            deflect_color = (156, 163, 175)
            deflect_bg = (35, 35, 45)
            deflect_border = (75, 85, 99)

        status_rect = pygame.Rect(PANEL_X + 14, curr_y, PANEL_WIDTH - 28, 20)
        pygame.draw.rect(self.screen, deflect_bg, status_rect)
        pygame.draw.rect(self.screen, deflect_border, status_rect, 1)

        d_surf = self.font_small.render(deflect_text, True, deflect_color)
        self.screen.blit(d_surf, (PANEL_X + 22, curr_y + 3))

        # Reliability Metrics: Actual Collisions & Measured Avoidance Events
        metric_str = f"Collisions: {total_collisions} | Avoided: {total_avoidances}"
        m_color = COLOR_HIGH_RISK if total_collisions > 0 else (209, 213, 219)
        m_surf = self.font_small.render(metric_str, True, m_color)
        self.screen.blit(m_surf, (PANEL_X + PANEL_WIDTH - m_surf.get_width() - 22, curr_y + 3))
        curr_y += 24

        # Simulation Clock & Recording Status Pill
        rec_active = stats.get("is_recording", False)
        rec_dot_color = (239, 68, 68) if rec_active else (107, 114, 128)
        # Pulse dot when recording
        if rec_active and int(self.anim_time * 2) % 2 == 0:
            rec_dot_color = (255, 120, 120)

        pygame.draw.circle(self.screen, rec_dot_color, (PANEL_X + 22, curr_y + 8), 5)
        status_text = f"REC: {'ACTIVE' if rec_active else 'PAUSED'}  |  {stats.get('total_records', 0):,} rows"
        rec_surf = self.font_small.render(status_text, True, COLOR_TEXT_PRIMARY if rec_active else COLOR_TEXT_MUTED)
        self.screen.blit(rec_surf, (PANEL_X + 34, curr_y + 2))

        # Time & FPS
        time_fps_str = f"Time: {sim_time:.1f}s | FPS: {int(fps)} {'[PAUSED]' if is_paused else ''}"
        tf_surf = self.font_small.render(time_fps_str, True, (156, 163, 175))
        self.screen.blit(tf_surf, (PANEL_X + PANEL_WIDTH - tf_surf.get_width() - 14, curr_y + 2))
        curr_y += 20

        # Risk Distribution Mini Bar
        total_recs = max(1, stats.get("total_records", 0))
        safe_p = stats.get("safe_count", 0) / total_recs
        warn_p = stats.get("warning_count", 0) / total_recs
        crit_p = stats.get("high_risk_count", 0) / total_recs

        bar_w = PANEL_WIDTH - 28
        bar_h = 5
        bar_x = PANEL_X + 14
        pygame.draw.rect(self.screen, (35, 45, 60), (bar_x, curr_y, bar_w, bar_h))

        w_safe = int(bar_w * safe_p)
        w_warn = int(bar_w * warn_p)
        w_crit = bar_w - w_safe - w_warn

        if w_safe > 0:
            pygame.draw.rect(self.screen, COLOR_SAFE, (bar_x, curr_y, w_safe, bar_h))
        if w_warn > 0:
            pygame.draw.rect(self.screen, COLOR_WARNING, (bar_x + w_safe, curr_y, w_warn, bar_h))
        if w_crit > 0:
            pygame.draw.rect(self.screen, COLOR_HIGH_RISK, (bar_x + w_safe + w_warn, curr_y, w_crit, bar_h))
        curr_y += 8

        dist_label = f"Dataset Balance: Safe {safe_p*100:.0f}% | Warn {warn_p*100:.0f}% | High {crit_p*100:.0f}%"
        dist_surf = self.font_small.render(dist_label, True, (120, 135, 155))
        self.screen.blit(dist_surf, (PANEL_X + 14, curr_y))
        curr_y += 16

        # Section Divider
        pygame.draw.line(self.screen, (31, 41, 55), (PANEL_X + 14, curr_y), (PANEL_X + PANEL_WIDTH - 14, curr_y), 1)
        curr_y += 6

        # Section Header: Live Fleet Telemetry
        fleet_hdr = self.font_body_bold.render("ACTIVE ROBOT FLEET TELEMETRY", True, (186, 230, 253))
        self.screen.blit(fleet_hdr, (PANEL_X + 14, curr_y))
        curr_y += 16

        # Render Robot Telemetry Cards
        card_h = 74
        for r in robots:
            card_rect = pygame.Rect(PANEL_X + 14, curr_y, PANEL_WIDTH - 28, card_h)
            pygame.draw.rect(self.screen, COLOR_PANEL_CARD, card_rect)
            pygame.draw.rect(self.screen, (40, 52, 70), card_rect, 1)

            # Left color accent strip
            pygame.draw.rect(self.screen, r.color, (PANEL_X + 14, curr_y, 4, card_h))

            # Row 1: Robot Name & Risk/Deflection Badge
            r_title = f"{r.name} (ID: {r.id})"
            name_surf = self.font_body_bold.render(r_title, True, (255, 255, 255))
            self.screen.blit(name_surf, (PANEL_X + 26, curr_y + 5))

            # Deflection or Risk Badge Pill
            is_evading = getattr(r, "is_deflecting", False) and deflection_enabled
            if is_evading:
                badge_text = "DEFLECTING"
                badge_color = (0, 240, 255)
            else:
                badge_text = RISK_NAMES[r.risk_label]
                badge_color = COLOR_HIGH_RISK if r.risk_label == RISK_LABEL_HIGH_RISK else (
                    COLOR_WARNING if r.risk_label == RISK_LABEL_WARNING else COLOR_SAFE
                )

            badge_surf = self.font_small.render(badge_text, True, badge_color)
            badge_w = badge_surf.get_width() + 10
            badge_rect = pygame.Rect(PANEL_X + PANEL_WIDTH - 14 - badge_w - 6, curr_y + 4, badge_w, 16)
            pygame.draw.rect(self.screen, (15, 22, 33), badge_rect)
            pygame.draw.rect(self.screen, badge_color, badge_rect, 1)
            self.screen.blit(badge_surf, (badge_rect.x + 5, badge_rect.y + 1))

            # Row 2: Live Coordinates & Velocity
            coords_str = f"Pos: ({r.x:5.1f}, {r.y:5.1f})   Vel: ({r.vx:+5.1f}, {r.vy:+5.1f})"
            coords_surf = self.font_small.render(coords_str, True, (209, 213, 219))
            self.screen.blit(coords_surf, (PANEL_X + 26, curr_y + 22))

            # Row 3: Heading & Speed
            heading_info = f"Head: {r.compass_direction} {r.heading_deg:4.0f}°   Spd: {r.speed:4.1f} px/s"
            head_surf = self.font_small.render(heading_info, True, (156, 163, 175))
            self.screen.blit(head_surf, (PANEL_X + 26, curr_y + 38))

            # Row 4: TTC & Deflection Guidance
            if is_evading:
                steer_deg = getattr(r, "deflection_angle_deg", 0.0)
                blend_pct = getattr(r, "deflection_blend", 0.0) * 100.0
                ttc_str = f"EVADING: Steer {steer_deg:.0f}° | Blend {blend_pct:.0f}%"
                ttc_color = (0, 240, 255)
            elif r.nearest_threat_id is not None and not math.isinf(r.min_ttc):
                ttc_str = f"Min TTC: {r.min_ttc:4.2f}s [vs Robot-{r.nearest_threat_id}]"
                ttc_color = badge_color
            else:
                ttc_str = "Min TTC: >3.00s (Safe Clear Path)"
                ttc_color = COLOR_SAFE

            ttc_surf = self.font_small.render(ttc_str, True, ttc_color)
            self.screen.blit(ttc_surf, (PANEL_X + 26, curr_y + 54))

            curr_y += card_h + 5

        # Active Risk & Interaction Summary
        curr_y += 2
        pygame.draw.line(self.screen, (31, 41, 55), (PANEL_X + 14, curr_y), (PANEL_X + PANEL_WIDTH - 14, curr_y), 1)
        curr_y += 6

        # Active Pairwise Warnings
        active_threats = [s for s in pairwise_summaries if s["risk_label"] != RISK_LABEL_SAFE]
        threat_hdr = f"ACTIVE CONVERGENCES ({len(active_threats)})"
        th_surf = self.font_body_bold.render(threat_hdr, True, (254, 215, 170))
        self.screen.blit(th_surf, (PANEL_X + 14, curr_y))
        curr_y += 16

        if active_threats:
            for s in active_threats[:3]:  # Show top 3 active conflicts
                p_text = f"Pair R{s['pair'][0]} <-> R{s['pair'][1]}: TTC {s['ttc']:.2f}s  Dist {s['dist']:.1f}px"
                c = COLOR_HIGH_RISK if s["risk_label"] == RISK_LABEL_HIGH_RISK else COLOR_WARNING
                row_surf = self.font_small.render(p_text, True, c)
                self.screen.blit(row_surf, (PANEL_X + 16, curr_y))
                curr_y += 14
        else:
            no_th_surf = self.font_small.render("All pairwise trajectories currently clear (Safe).", True, (100, 116, 139))
            self.screen.blit(no_th_surf, (PANEL_X + 16, curr_y))
            curr_y += 14

        # Footer Keyboard Controls Quick Reference
        help_bg = pygame.Rect(PANEL_X + 14, PANEL_Y + PANEL_HEIGHT - 64, PANEL_WIDTH - 28, 52)
        pygame.draw.rect(self.screen, (15, 23, 42), help_bg)
        pygame.draw.rect(self.screen, (30, 41, 59), help_bg, 1)

        k1 = self.font_small.render("[SPACE] Pause   [D] Deflection Toggle   [R] REC Toggle", True, (148, 163, 184))
        k2 = self.font_small.render("[1-4] Scenarios   [N] New Targets   [C] CSV   [ESC] Quit", True, (148, 163, 184))
        self.screen.blit(k1, (help_bg.x + 8, help_bg.y + 10))
        self.screen.blit(k2, (help_bg.x + 8, help_bg.y + 28))
