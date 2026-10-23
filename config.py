"""
Configuration and Global Constants for Multi-Robot Collision-Risk Simulation.
Phase 1: Advanced UI & Data Generation Engine.
"""

from dataclasses import dataclass
from typing import Tuple, List

# ==============================================================================
# DISPLAY & WINDOW SETTINGS
# ==============================================================================
WINDOW_WIDTH = 1320
WINDOW_HEIGHT = 800
FPS = 60

# Arena (Bounded Room) Viewport Settings
ARENA_X = 20
ARENA_Y = 20
ARENA_WIDTH = 880
ARENA_HEIGHT = 760

# Side Panel (Telemetry & Real-Time Monitoring) Settings
PANEL_X = 920
PANEL_Y = 20
PANEL_WIDTH = 380
PANEL_HEIGHT = 760

# ==============================================================================
# COLOR PALETTE (Futuristic Cyber-Dark Aesthetic)
# ==============================================================================
COLOR_BG_DARK = (10, 14, 22)          # Deep cosmic dark
COLOR_PANEL_BG = (17, 24, 39)         # Dark slate sidebar
COLOR_PANEL_BORDER = (31, 41, 55)     # Subtle border
COLOR_PANEL_CARD = (24, 33, 47)       # Telemetry card background
COLOR_ARENA_BG = (13, 17, 26)         # Arena floor
COLOR_ARENA_BORDER = (0, 210, 255)    # Neon cyan arena bounds
COLOR_ARENA_GRID = (22, 30, 44)       # Subtle floor grid
COLOR_ARENA_ACCENT = (40, 56, 80)     # Grid sub-accents

# Text & Interface Colors
COLOR_TEXT_PRIMARY = (243, 244, 246)
COLOR_TEXT_MUTED = (156, 163, 175)
COLOR_TEXT_ACCENT = (56, 189, 248)    # Sky blue

# Risk State Colors
COLOR_SAFE = (16, 185, 129)           # Emerald Green (TTC > 3.0s)
COLOR_WARNING = (245, 158, 11)        # Radiant Amber (1.0s < TTC <= 3.0s)
COLOR_HIGH_RISK = (239, 68, 68)       # Bright Crimson / Red (TTC <= 1.0s)

# Distinct Robot Colors (5-Robot Fleet)
ROBOT_COLORS: List[Tuple[int, int, int]] = [
    (0, 240, 255),    # Robot 0: Neon Cyan
    (255, 107, 74),   # Robot 1: Coral Blaze
    (52, 211, 153),   # Robot 2: Mint Emerald
    (192, 132, 252),  # Robot 3: Electric Violet
    (251, 191, 36),   # Robot 4: Sunburst Amber
]

# Static Obstacle Colors
COLOR_OBSTACLE_FILL = (30, 41, 59)
COLOR_OBSTACLE_BORDER = (71, 85, 105)
COLOR_OBSTACLE_HAZARD = (245, 158, 11)

# ==============================================================================
# ROBOT KINEMATICS & PHYSICS SETTINGS
# ==============================================================================
DEFAULT_ROBOT_COUNT = 5
ROBOT_RADIUS = 18.0                   # Visual and collision radius in pixels
ROBOT_MAX_SPEED = 120.0               # Pixels per second
ROBOT_MIN_SPEED = 30.0                # Pixels per second
ROBOT_MAX_ACCEL = 160.0               # Pixels per second squared
GOAL_REACH_DISTANCE = 22.0            # Distance at which goal is considered reached
WALL_BOUNCE_RESTITUTION = 0.85        # Elasticity when contacting arena walls
ROBOT_COLLISION_RESTITUTION = 0.90    # Elasticity during inter-robot contact

# ==============================================================================
# RISK CLASSIFICATION & TIME-TO-COLLISION (TTC) THRESHOLDS
# ==============================================================================
TTC_SAFE_THRESHOLD = 3.0              # TTC > 3.0s => 0 (Safe)
TTC_WARNING_THRESHOLD = 1.0           # 1.0s < TTC <= 3.0s => 1 (Warning)
                                      # TTC <= 1.0s => 2 (High Risk)
TTC_MAX_CLIP = 99.9                   # Numerical cap for infinite TTC in logs

# Risk Labels
RISK_LABEL_SAFE = 0
RISK_LABEL_WARNING = 1
RISK_LABEL_HIGH_RISK = 2

RISK_NAMES = {
    RISK_LABEL_SAFE: "SAFE",
    RISK_LABEL_WARNING: "WARNING",
    RISK_LABEL_HIGH_RISK: "HIGH_RISK",
}

# ==============================================================================
# DATA LOGGER CONFIGURATION
# ==============================================================================
DEFAULT_DATASET_DIR = "dataset"
DEFAULT_DATASET_FILENAME = "collision_risk_dataset.csv"
DATA_LOG_INTERVAL_FRAMES = 2          # Log every 2 frames (~30Hz logging at 60 FPS)
AUTO_FLUSH_INTERVAL = 100             # Rows between disk flushes

# ==============================================================================
# PHASE 4: ADAPTIVE DEFLECTION CONFIGURATION (STAGE C)
# ==============================================================================
DEFLECTION_HYSTERESIS_SECONDS = 0.40   # Minimum persistence time in deflection state
DEFLECTION_RECOVERY_SECONDS = 0.35     # Decay duration to blend smoothly back to goal
DEFLECTION_HORIZON_SECONDS = 1.5       # Forward projection time horizon for rollouts
DEFLECTION_PROJECTION_STEPS = 5        # Discretization steps along rollout trajectory
DEFLECTION_ROBOT_SAFETY_MARGIN = 16.0  # Buffer distance in pixels beyond 2 * ROBOT_RADIUS
DEFLECTION_OBSTACLE_SAFETY_MARGIN = 18.0 # Buffer distance in pixels beyond obs.radius + ROBOT_RADIUS
DEFLECTION_WALL_SAFETY_MARGIN = 24.0   # Buffer distance in pixels from arena boundaries
DEFLECTION_HIGH_RISK_PROB_THRESH = 0.45 # GNN high-risk probability activation threshold

