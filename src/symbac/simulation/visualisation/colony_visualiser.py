import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import colorsys
import typing
if typing.TYPE_CHECKING:
    from symbac.simulation.simcell import SimCell

# from your_utility_module import get_opposite_color

class ColonyVisualiser:
    """
    Manages the state of the viewport (zoom and center) and draws the colony.
    """

    def __init__(self, initial_view_range: float = 400.0, zoom_level: float = 1.5, fill_threshold: float = 0.9):
        """
        Initializes the visualizer.

        Parameters
        ----------
        initial_view_range : float
            The initial width and height of the viewport.
        zoom_level : float
            How much to zoom out when the threshold is reached (e.g., 1.5 means 50% larger view).
        fill_threshold : float
            The percentage (0.0 to 1.0) of the view that can be filled before zooming out.
        """
        self.view_range = initial_view_range
        self.view_center = np.array([0.0, 0.0])  # Start centered at the origin
        self.zoom_level = zoom_level
        self.fill_threshold = fill_threshold
        self.frame_number = 0


    # @staticmethod
    # def get_daughter_colour(cell: 'SimCell', next_group_id: int) -> tuple[int, int, int]:
    #     # 1. Get the mother's color and normalize it to the 0-1 range for colorsys
    #     r, g, b = cell.base_color
    #     r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    #
    #     # 2. Convert RGB to HSV
    #     h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    #     # 3. Mutate the Hue to change the color while preserving lineage
    #     #    A small hue shift changes the color along the color wheel (e.g., red -> orange)
    #     hue_shift = np.random.uniform(-1, 1) / (np.sqrt(next_group_id) * 2)  # Shift hue with biased rw
    #     #s_shift = np.random.uniform(-0.2, 0.2) / (np.sqrt(next_group_id) / 1.8)  # Slight saturation shift
    #     #v_shift = np.random.uniform(-0.2, 0.2) / (np.sqrt(next_group_id) / 1.8) # Slight brightness shift
    #     new_h = (h + hue_shift) % 1.0  # Use modulo to wrap around the color wheel
    #     #    This prevents colors from becoming grayish or dark.
    #     #    We'll clamp them to a minimum vibrancy level.
    #     new_s = s
    #     new_v = v
    #
    #     # 5. Convert the new HSV color back to RGB
    #     new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, new_s, new_v)
    #
    #     # 6. Scale back to 0-255 and create the final tuple
    #     daughter_color = (int(new_r * 255), int(new_g * 255), int(new_b * 255))
    #     return daughter_color

    # @staticmethod
    # def update_colors(cell) -> None:
    #     if not cell.PhysicsRepresentation.segments: return
    #     a = 255
    #     r, g, b = cell.base_color
    #
    #     r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    #
    #     # 2. Convert RGB to HSV
    #     h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    #     new_s = max(s / np.sqrt(cell.num_divisions+1), 0.3)  # Ensure saturation is not too low
    #     new_v = max(v / np.sqrt(cell.num_divisions+1), 0.3) # Ensure brightness is not too low
    #
    #     # 5. Convert the new HSV color back to RGB
    #     r, g, b = colorsys.hsv_to_rgb(h, new_s, new_v)
    #     r, g, b = (int(r * 255), int(g * 255), int(b * 255))
    #
    #     body_color = (r,g,b,a)
    #     head_color = (min(255, int(r * 1.3)), min(255, int(g * 1.3)), min(255, int(b * 1.3)), a)
    #     tail_color = (int(r * 0.7), int(g * 0.7), int(b * 0.7), a)
    #     for segment in cell.PhysicsRepresentation.segments: # You have to set a color attribute for pygame
    #         segment.shape.color = body_color
    #     cell.PhysicsRepresentation.segments[0].shape.color = head_color
    #     cell.PhysicsRepresentation.segments[-1].shape.color = tail_color
