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


def _draw_thick_segment(image, p1, p2, thickness, value):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    radius = max(1.0, float(thickness))
    distance = float(np.hypot(x2 - x1, y2 - y1))
    steps = max(1, int(np.ceil(distance / max(1.0, radius * 0.5))))
    for idx in range(steps + 1):
        alpha = idx / max(steps, 1)
        x = x1 + alpha * (x2 - x1)
        y = y1 + alpha * (y2 - y1)
        rr, cc = disk((y, x), radius, shape=image.shape)
        image[rr, cc] = value


def render_live_frame_image(frame_segments, static_segments, scene_shape, wall_value=96):
    """Rasterize exact geometry segments plus current cells into a preview image."""
    image = np.zeros(scene_shape, dtype=np.uint8)

    for segment in static_segments:
        _draw_thick_segment(image, segment.p1, segment.p2, segment.thickness, wall_value)

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
                distance = float(np.hypot(px - prev_position[0], py - prev_position[1]))
                step = max(1.0, min(pr, prev_radius))
                n_interp = max(1, int(np.ceil(distance / step)))
                for idx in range(1, n_interp):
                    alpha = idx / n_interp
                    ix = prev_position[0] + alpha * (px - prev_position[0])
                    iy = prev_position[1] + alpha * (py - prev_position[1])
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
