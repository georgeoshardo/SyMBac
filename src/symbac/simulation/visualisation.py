import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

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

    def draw_colony_matplotlib(self, colony: list[Cell], output_dir: str = "frames"):
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
                    linewidth=1.0
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