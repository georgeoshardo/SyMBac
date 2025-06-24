import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import colorsys
import typing
if typing.TYPE_CHECKING:
    from symbac.simulation.cell import Cell

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

    def draw_colony_matplotlib(self, colony: list['Cell'], output_dir: str = "frames"):
        """
        Draws the current state of the colony, updating the zoom level if the colony
        fills more than the specified threshold of the viewport.
        """
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 10))
        # --- New Zoom Logic ---

        # Only perform calculations if the colony is not empty
        if any(cell.segments for cell in colony):
            # 1. Calculate the colony's current bounding box
            all_positions = np.array([
                seg.body.position for cell in colony for seg in cell.segments
            ])
            min_coords = all_positions.min(axis=0)
            max_coords = all_positions.max(axis=0)

            colony_width = max_coords[0] - min_coords[0]
            colony_height = max_coords[1] - min_coords[1]

            # 2. Check if the colony has grown beyond the fill threshold
            if (colony_width > self.view_range * self.fill_threshold or
                    colony_height > self.view_range * self.fill_threshold):
                print(f"Frame {self.frame_number}: Zoom threshold reached. Rescaling view.")

                # 3. If so, update the view range to fit the entire colony plus a buffer
                self.view_range = max(colony_width, colony_height) * self.zoom_level

                # 4. Re-center the view on the colony's new center
                self.view_center = (min_coords + max_coords) / 2

        # 5. Set plot limits based on the current view state
        ax.set_xlim(self.view_center[0] - self.view_range / 2, self.view_center[0] + self.view_range / 2)
        ax.set_ylim(self.view_center[1] + self.view_range / 2, self.view_center[1] - self.view_range / 2)

        # --- Drawing Logic (Unchanged) ---
        for cell in colony:
            for segment in cell.segments:
                x, y = segment.body.position
                r = segment.shape.radius

                rgba_fill_color = np.array(segment.shape.color) / 255.0

                # Using a placeholder for the opposite color function
                # rgba_border_color = get_opposite_color(rgba_fill_color)
                rgba_border_color = 'white'

                circle = patches.Circle(
                    (x, y),
                    radius=r,
                    facecolor=rgba_fill_color,
                    edgecolor=rgba_border_color,
                    linewidth=1.0,
                    alpha=0.25
                )
                ax.add_patch(circle)

        # --- File Saving (Unchanged) ---
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, f"frame_{self.frame_number:04d}.jpeg")
        plt.axis('off')
        plt.tight_layout()

        plt.savefig(filename)
        plt.close(fig)
        self.frame_number += 1


    @staticmethod
    def get_daughter_colour(cell: 'Cell', next_group_id: int) -> tuple[int, int, int]:
        # 1. Get the mother's color and normalize it to the 0-1 range for colorsys
        r, g, b = cell.base_color
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0

        # 2. Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        # 3. Mutate the Hue to change the color while preserving lineage
        #    A small hue shift changes the color along the color wheel (e.g., red -> orange)
        hue_shift = np.random.uniform(-1, 1) / (np.sqrt(next_group_id) * 2)  # Shift hue with biased rw
        #s_shift = np.random.uniform(-0.2, 0.2) / (np.sqrt(next_group_id) / 1.8)  # Slight saturation shift
        #v_shift = np.random.uniform(-0.2, 0.2) / (np.sqrt(next_group_id) / 1.8) # Slight brightness shift
        new_h = (h + hue_shift) % 1.0  # Use modulo to wrap around the color wheel
        #    This prevents colors from becoming grayish or dark.
        #    We'll clamp them to a minimum vibrancy level.
        new_s = s
        new_v = v

        # 5. Convert the new HSV color back to RGB
        new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, new_s, new_v)

        # 6. Scale back to 0-255 and create the final tuple
        daughter_color = (int(new_r * 255), int(new_g * 255), int(new_b * 255))
        return daughter_color

    @staticmethod
    def update_colors(cell) -> None:
        if not cell.PhysicsRepresentation.segments: return
        a = 255
        r, g, b = cell.base_color

        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0

        # 2. Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        new_s = max(s / np.sqrt(cell.num_divisions+1), 0.3)  # Ensure saturation is not too low
        new_v = max(v / np.sqrt(cell.num_divisions+1), 0.3) # Ensure brightness is not too low

        # 5. Convert the new HSV color back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, new_s, new_v)
        r, g, b = (int(r * 255), int(g * 255), int(b * 255))

        body_color = (r,g,b,a)
        head_color = (min(255, int(r * 1.3)), min(255, int(g * 1.3)), min(255, int(b * 1.3)), a)
        tail_color = (int(r * 0.7), int(g * 0.7), int(b * 0.7), a)
        for segment in cell.PhysicsRepresentation.segments:
            segment.shape.color = body_color
        cell.PhysicsRepresentation.segments[0].shape.color = head_color
        cell.PhysicsRepresentation.segments[-1].shape.color = tail_color
