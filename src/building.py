import numpy as np
import pyvista as pv
from src.settings import load_config

CONFIG = load_config()
WIDTH = float(CONFIG["building"]["width"])
DEPTH = float(CONFIG["building"]["depth"])
FLOOR_HEIGHT = float(CONFIG["building"]["floor_height"])
FLOORS = int(CONFIG["building"]["floors"])
TOTAL_HEIGHT = FLOOR_HEIGHT * FLOORS

# Open space in the middle (x)
COURTYARD = {"x1": 7.0, "x2": 11.0, "y1": 4.5, "y2": 10.0}

plotter = pv.Plotter(window_size=CONFIG["visualization"]["window_size"])
plotter.set_background(CONFIG["visualization"]["background"])
LINE_COLOR = CONFIG["visualization"]["wall_color"]
LINE_WIDTH = 2

# Boarding house (kos) building
def draw_line(p1, p2, color=LINE_COLOR, width=LINE_WIDTH):
    points = np.array([p1, p2], dtype=float)
    plotter.add_lines(
        points,
        color=color,
        width=width,
        connected=False
    )

def horizontal_line(x1, y1, x2, y2, z):
    draw_line((x1, y1, z), (x2, y2, z))

def vertical_line(x, y, z1, z2):
    draw_line((x, y, z1), (x, y, z2))

def wall_x(x1, x2, y, z):
    horizontal_line(x1, y, x2, y, z)

def wall_y(x, y1, y2, z):
    horizontal_line(x, y1, x, y2, z)

def draw_door_x(x, y, z, width=0.9):
    horizontal_line(x, y, x + width, y, z)
    vertical_line(x, y, z, z + 0.05)

# Windows
def draw_window_x(x, y, z, width=1.0):
    height = 0.7
    horizontal_line(x, y, x + width, y, z)
    horizontal_line(
        x, y, x + width, y, z + height
    )
    vertical_line(x, y, z, z + height)
    vertical_line(
        x + width, y, z, z + height
    )

# 3-Floor Building Plan
def draw_floor_plan(z, floor_number):
    # 1. Exterior walls
    if floor_number == 1: # Enterance door
        wall_x(0, 7.0, 0, z)
        wall_x(9.0, WIDTH, 0, z)
    else:
        wall_x(0, WIDTH, 0, z)

    wall_x(0, WIDTH, DEPTH, z)
    wall_y(0, 0, DEPTH, z)
    wall_y(WIDTH, 0, DEPTH, z)

    # 2. Rear Room
    wall_x(0, 6.0, 10.0, z)
    wall_x(6.0, 7.0, 10.0, z)
    wall_x(11.0, 12.0, 10.0, z)
    wall_x(12.0, WIDTH, 10.0, z)

    # Partition
    wall_y(6.0, 10.0, DEPTH, z)
    wall_y(12.0, 10.0, DEPTH, z)

    # 3. Corridor and front room 
    wall_x(0, 2.0, 4.5, z)
    wall_x(2.9, 6.0, 4.5, z)
    wall_x(6.9, 7.0, 4.5, z)
    wall_x(11.0, 12.0, 4.5, z)
    wall_x(12.9, 13.0, 4.5, z)
    wall_x(13.9, WIDTH, 4.5, z)

    # Partition
    wall_y(6.0, 0, 4.5, z)
    wall_y(12.0, 0, 4.5, z)
    wall_y(13.0, 0, 4.5, z)

    # 4. Open space
    x1 = COURTYARD["x1"]
    x2 = COURTYARD["x2"]
    y1 = COURTYARD["y1"]
    y2 = COURTYARD["y2"]

    wall_y(x1, y1, y2, z)
    wall_y(x2, y1, y2, z)
    wall_x(x1, x2, y1, z)
    wall_x(x1, x2, y2, z)

    # 5. Staircase area
    stair_x1, stair_x2 = 0.0, 4.8
    stair_y1, stair_y2 = 5.0, 6.3

    wall_x(stair_x1, stair_x2, stair_y1, z)
    wall_x(stair_x1, stair_x2, stair_y2, z)
    number_steps = 6

    for i in range(1, number_steps):
        x = stair_x1 + (
            (stair_x2 - stair_x1) * i / number_steps
        )
        wall_y(x, stair_y1, stair_y2, z)
    
    # Rare room windows
    draw_window_x(2.0, 10.0, z, 1.0)
    draw_window_x(8.0, 10.0, z, 1.0)
    draw_window_x(14.0, 10.0, z, 1.0)

    # Front room windows
    draw_window_x(3.0, 4.5, z, 1.0)
    draw_window_x(8.0, 4.5, z, 1.0)
    draw_window_x(14.5, 4.5, z, 1.0)

    # Door
    draw_door_x(2.0, 4.5, z, 0.9)
    draw_door_x(6.0, 10.0, z, 0.9)
    draw_door_x(11.0, 10.0, z, 0.9)
    draw_door_x(12.0, 4.5, z, 0.9)


# 3-Floor Building
for floor in range(FLOORS):
    z = floor * FLOOR_HEIGHT
    draw_floor_plan(z, floor + 1)

# Corner of Building
outer_corners = [
    (0, 0), (WIDTH, 0),
    (WIDTH, DEPTH), (0, DEPTH)
]
for x, y in outer_corners:
    vertical_line(x, y, 0, TOTAL_HEIGHT)

# Corner of Courtyard (x)
x1 = COURTYARD["x1"]
x2 = COURTYARD["x2"]
y1 = COURTYARD["y1"]
y2 = COURTYARD["y2"]

for x, y in [
    (x1, y1), (x2, y1),
    (x1, y2), (x2, y2)
]:
    vertical_line(x, y, 0, TOTAL_HEIGHT)

# Primary partition walls
for x in [6.0, 12.0, 13.0]:
    for y in [0.0, 4.5, 10.0, DEPTH]:
        vertical_line(x, y, 0, TOTAL_HEIGHT)

# Staircase corner
for x, y in [(0, 5.0), (4.8, 5.0),
             (0, 6.3), (4.8, 6.3)]:
    vertical_line(x, y, 0, TOTAL_HEIGHT)

# X for open space (courtyard)
for floor in range(FLOORS):
    z = floor * FLOOR_HEIGHT + 0.1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    size = 0.25

    draw_line(
        (cx - size, cy - size, z),
        (cx + size, cy + size, z)
    )
    draw_line(
        (cx - size, cy + size, z),
        (cx + size, cy - size, z)
    )

# Visualization
plotter.add_text(
    "KOS 3 LANTAI | WIREFRAME",
    position="upper_left",
    color="white",
    font_size=14
)

plotter.add_axes(
    line_width=1,
    color="white"
)

plotter.show_grid(
    color="gray",
    grid="back",
    location="outer"
)

plotter.camera_position = "iso"
plotter.camera.zoom(0.85)
