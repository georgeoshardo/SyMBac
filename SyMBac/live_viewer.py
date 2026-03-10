import numpy as np
from skimage.draw import disk


def build_live_frame_segments(cells):
    """Extract segment-chain geometry for the live viewer."""
    frame_segments = []
    for cell in cells:
        physics_rep = getattr(cell, "physics_representation", None)
        segments = getattr(physics_rep, "segments", None)
        if not segments:
            continue
        positions = np.array(
            [[float(segment.position[0]), float(segment.position[1])] for segment in segments],
            dtype=np.float64,
        )
        radii = np.array(
            [float(getattr(segment, "radius", 0.0)) for segment in segments],
            dtype=np.float64,
        )
        frame_segments.append((positions, radii))
    return frame_segments


def render_live_frame_image(
    frame_segments,
    scene_shape,
    trench_center_x,
    trench_width,
    trench_height,
    wall_thickness=3,
):
    """Rasterize the current segment-chain state into a preview image."""
    image = np.zeros(scene_shape, dtype=np.uint8)
    height, width = image.shape

    half_width = float(trench_width) / 2.0
    x_left = int(round(trench_center_x - half_width))
    x_right = int(round(trench_center_x + half_width))
    y_side_start = int(round(half_width))
    y_top = int(round(trench_height))

    wall_value = 96

    x0 = max(0, x_left - wall_thickness)
    x1 = min(width, x_left + wall_thickness + 1)
    x2 = max(0, x_right - wall_thickness)
    x3 = min(width, x_right + wall_thickness + 1)
    y0 = max(0, y_side_start)
    y1 = min(height, y_top)

    if y1 > y0:
        image[y0:y1, x0:x1] = wall_value
        image[y0:y1, x2:x3] = wall_value

    arc_samples = max(32, int(np.ceil(np.pi * max(half_width, 1.0))))
    arc_xs = np.linspace(-half_width, half_width, arc_samples)
    arc_ys = half_width - np.sqrt(np.maximum(0.0, half_width**2 - arc_xs**2))
    for dx, dy in zip(arc_xs, arc_ys):
        rr, cc = disk(
            (dy, trench_center_x + dx),
            max(1.0, float(wall_thickness)),
            shape=image.shape,
        )
        image[rr, cc] = wall_value

    for positions, radii in frame_segments:
        if len(positions) == 0:
            continue

        prev_position = None
        prev_radius = None
        for position, radius in zip(positions, radii):
            px = float(position[0])
            py = float(position[1])
            pr = max(1.0, float(radius))
            rr, cc = disk((py, px), pr, shape=image.shape)
            image[rr, cc] = 255

            if prev_position is not None:
                dx = px - prev_position[0]
                dy = py - prev_position[1]
                distance = float(np.hypot(dx, dy))
                step = max(1.0, min(pr, prev_radius))
                n_interp = max(1, int(np.ceil(distance / step)))
                for idx in range(1, n_interp):
                    alpha = idx / n_interp
                    ix = prev_position[0] + alpha * dx
                    iy = prev_position[1] + alpha * dy
                    ir = max(1.0, prev_radius + alpha * (pr - prev_radius))
                    rr, cc = disk((iy, ix), ir, shape=image.shape)
                    image[rr, cc] = 255

            prev_position = (px, py)
            prev_radius = pr

    return image


class LiveSimulationViewer:
    """Minimal pyqtgraph-based live viewer for the simulation preview."""

    def __init__(self, scene_shape, title="SyMBac Live", window_size=(900, 900)):
        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtCore, QtWidgets
        except ImportError as e:
            raise ImportError(
                "show_window=True requires pyqtgraph and a working Qt binding. "
                "Install pyqtgraph together with PyQt5/PyQt6 or PySide6."
            ) from e

        self.pg = pg
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.app = pg.mkQApp("SyMBac")

        self.window = pg.GraphicsLayoutWidget(title=title, show=False)
        self.window.resize(*window_size)
        self.window.setWindowTitle(title)

        self.plot_item = self.window.addPlot()
        self.plot_item.hideAxis("left")
        self.plot_item.hideAxis("bottom")
        self.plot_item.setAspectLocked(True)
        self.plot_item.invertY(True)
        self.plot_item.setMouseEnabled(x=False, y=False)
        self.plot_item.setMenuEnabled(False)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot_item.addItem(self.image_item)
        self.plot_item.setRange(
            xRange=(0, scene_shape[1]),
            yRange=(scene_shape[0], 0),
            padding=0.02,
        )

        self._closed = False
        original_close_event = self.window.closeEvent
        original_key_press_event = self.window.keyPressEvent

        def close_event(event):
            self._closed = True
            original_close_event(event)

        def key_press_event(event):
            if event.key() == self.QtCore.Qt.Key_Escape:
                self.window.close()
                event.accept()
                return
            original_key_press_event(event)

        self.window.closeEvent = close_event
        self.window.keyPressEvent = key_press_event
        self.window.show()
        self.process_events()

    @property
    def closed(self):
        return self._closed or not self.window.isVisible()

    def update_image(self, image, title=None):
        self.image_item.setImage(image, autoLevels=False)
        if title is not None:
            self.window.setWindowTitle(title)

    def process_events(self):
        self.app.processEvents()

    def close(self):
        if not self.closed:
            self.window.close()
        self.process_events()
