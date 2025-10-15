import numpy as np
import cv2
from typing import List

from .aot import Root
from .const import IMAGE_SIZE, DEFAULT_LINE_WIDTH


def _draw_triangle(img, center, size, color, thickness):
    cy, cx = center
    r = int(size * min(img.shape) / 2)
    pts = np.array([
        [cy - r, cx],
        [cy + r, cx - int(0.866 * r)],
        [cy + r, cx + int(0.866 * r)],
    ], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_square(img, center, size, color, thickness):
    cy, cx = center
    r = int(size * min(img.shape) / 2)
    pt1 = (cx - r, cy - r)
    pt2 = (cx + r, cy + r)
    cv2.rectangle(img, pt1, pt2, color, thickness)


def _draw_pentagon(img, center, size, color, thickness):
    cy, cx = center
    r = int(size * min(img.shape) / 2)
    pts = []
    for k in range(5):
        theta = -np.pi / 2 + 2 * np.pi * k / 5
        pts.append([cy + int(r * np.sin(theta)), cx + int(r * np.cos(theta))])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_hexagon(img, center, size, color, thickness):
    cy, cx = center
    r = int(size * min(img.shape) / 2)
    pts = []
    for k in range(6):
        theta = -np.pi / 2 + 2 * np.pi * k / 6
        pts.append([cy + int(r * np.sin(theta)), cx + int(r * np.cos(theta))])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_circle(img, center, size, color, thickness):
    cy, cx = center
    r = int(size * min(img.shape) / 2)
    cv2.circle(img, (cx, cy), r, color, thickness)


def render_panel(root: Root) -> np.ndarray:
    canvas = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) * 255
    layout = root.children[0].components[0].layout
    # draw entities in black lines on white background
    for ent in layout.entities:
        y, x, h, w = ent.bbox
        cy = int(y * IMAGE_SIZE)
        cx = int(x * IMAGE_SIZE)
        size = min(h, w)
        color = 0
        if ent.type.get_value() == "triangle":
            _draw_triangle(canvas, (cy, cx), size, color, DEFAULT_LINE_WIDTH)
        elif ent.type.get_value() == "square":
            _draw_square(canvas, (cy, cx), size, color, DEFAULT_LINE_WIDTH)
        elif ent.type.get_value() == "pentagon":
            _draw_pentagon(canvas, (cy, cx), size, color, DEFAULT_LINE_WIDTH)
        elif ent.type.get_value() == "hexagon":
            _draw_hexagon(canvas, (cy, cx), size, color, DEFAULT_LINE_WIDTH)
        else:
            _draw_circle(canvas, (cy, cx), size, color, DEFAULT_LINE_WIDTH)
    return canvas


def generate_matrix(array_list: List[np.ndarray]) -> np.ndarray:
    assert len(array_list) <= 9
    grid = np.ones((IMAGE_SIZE * 3, IMAGE_SIZE * 3), dtype=np.uint8) * 255
    for idx, panel in enumerate(array_list):
        i, j = divmod(idx, 3)
        grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE, j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = panel
    # grid lines
    k1 = int(IMAGE_SIZE * 0.33)
    k2 = int(IMAGE_SIZE * 0.67)
    grid[k1 - 1:k1 + 1, :] = 0
    grid[k2 - 1:k2 + 1, :] = 0
    grid[:, k1 - 1:k1 + 1] = 0
    grid[:, k2 - 1:k2 + 1] = 0
    return grid


