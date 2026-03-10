from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymunk


@dataclass(frozen=True)
class Bounds2D:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    def translated(self, dx: float, dy: float) -> "Bounds2D":
        return Bounds2D(
            min_x=self.min_x + dx,
            min_y=self.min_y + dy,
            max_x=self.max_x + dx,
            max_y=self.max_y + dy,
        )


@dataclass(frozen=True)
class SegmentPrimitive:
    p1: tuple[float, float]
    p2: tuple[float, float]
    thickness: float

    def translated(self, dx: float, dy: float) -> "SegmentPrimitive":
        return SegmentPrimitive(
            p1=(self.p1[0] + dx, self.p1[1] + dy),
            p2=(self.p2[0] + dx, self.p2[1] + dy),
            thickness=self.thickness,
        )


@dataclass(frozen=True)
class GeometryLayout:
    spec: "GeometrySpec"
    padding_x: float | None = None
    padding_y: float | None = None
    min_preview_size: int = 128

    def __post_init__(self):
        pad_x = float(self.padding_x if self.padding_x is not None else self.spec.default_padding_x)
        pad_y = float(self.padding_y if self.padding_y is not None else self.spec.default_padding_y)
        offset_x = pad_x - self.spec.local_bounds.min_x
        offset_y = pad_y - self.spec.local_bounds.min_y
        world_bounds = self.spec.local_bounds.translated(offset_x, offset_y)
        preview_shape = (
            max(self.min_preview_size, int(np.ceil(world_bounds.max_y + pad_y))),
            max(self.min_preview_size, int(np.ceil(world_bounds.max_x + pad_x))),
        )
        object.__setattr__(self, "padding_x", pad_x)
        object.__setattr__(self, "padding_y", pad_y)
        object.__setattr__(self, "world_offset", (offset_x, offset_y))
        object.__setattr__(self, "local_bounds", self.spec.local_bounds)
        object.__setattr__(self, "world_bounds", world_bounds)
        object.__setattr__(self, "preview_shape", preview_shape)

    def to_world_point(self, point: tuple[float, float]) -> tuple[float, float]:
        return (point[0] + self.world_offset[0], point[1] + self.world_offset[1])

    def to_local_point(self, point: tuple[float, float]) -> tuple[float, float]:
        return (point[0] - self.world_offset[0], point[1] - self.world_offset[1])

    def to_world_points(self, points):
        points = np.asarray(points, dtype=np.float64)
        return points + np.array(self.world_offset, dtype=np.float64)

    def to_local_points(self, points):
        points = np.asarray(points, dtype=np.float64)
        return points - np.array(self.world_offset, dtype=np.float64)


class GeometrySpec:
    @property
    def local_bounds(self) -> Bounds2D:
        raise NotImplementedError

    @property
    def default_padding_x(self) -> float:
        raise NotImplementedError

    @property
    def default_padding_y(self) -> float:
        raise NotImplementedError

    @property
    def characteristic_width(self) -> float:
        return self.local_bounds.width

    def build(self, space: pymunk.Space, layout: GeometryLayout) -> None:
        raise NotImplementedError

    def preview_primitives(self, layout: GeometryLayout) -> list[SegmentPrimitive]:
        raise NotImplementedError

    def seed_cell_local_position(self, segment_radius: float) -> tuple[float, float]:
        raise NotImplementedError

    def positions_within_bounds(
        self,
        positions,
        radii,
        layout: GeometryLayout,
        *,
        enforce_open_end_cap: bool,
    ) -> bool:
        raise NotImplementedError

    def cell_out_of_bounds(self, positions, radii, layout: GeometryLayout) -> bool:
        raise NotImplementedError

    def project_body_inside_bounds(self, body, radius: float, layout: GeometryLayout) -> tuple[bool, bool, bool]:
        raise NotImplementedError


def segment_creator(local_xy1, local_xy2, global_xy, thickness):
    segment_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment_shape = pymunk.Segment(segment_body, local_xy1, local_xy2, thickness)
    segment_body.position = global_xy
    segment_shape.friction = 0
    return segment_body, segment_shape


def _add_world_segment(space: pymunk.Space, primitive: SegmentPrimitive) -> None:
    body, shape = segment_creator(primitive.p1, primitive.p2, (0.0, 0.0), primitive.thickness)
    space.add(body, shape)


def semi_circle(r, x):
    return np.sqrt(r**2 - x**2)


def semi_circle_grad(r, x):
    x_array = np.asarray(x, dtype=np.float64)
    denominator = np.sqrt(np.clip(r**2 - x_array**2, 0.0, None))
    gradient = np.empty_like(x_array, dtype=np.float64)
    np.divide(-x_array, denominator, out=gradient, where=denominator > 0)
    endpoint_gradient = np.where(x_array < 0, np.inf, np.where(x_array > 0, -np.inf, 0.0))
    gradient = np.where(denominator > 0, gradient, endpoint_gradient)
    if np.isscalar(x):
        return float(gradient)
    return gradient


def dx(r, x, distance):
    return distance * np.cos(np.arctan(semi_circle_grad(r, x)))


def dy(r, x, distance):
    return distance * np.sin(np.arctan(semi_circle_grad(r, x)))


class TrenchGeometrySpec(GeometrySpec):
    def __init__(self, width: float, trench_length: float, barrier_thickness: float = 10.0, arc_samples: int = 50):
        self.width = float(width)
        self.trench_length = float(trench_length)
        self.barrier_thickness = float(barrier_thickness)
        self.arc_samples = int(arc_samples)
        self._local_segments = self._build_local_segments()
        self._local_bounds = self._compute_local_bounds(self._local_segments)
        self._inner_half_width = self.width / 2.0
        self._open_end_y = self._inner_half_width + self.trench_length
        default_padding = max(24.0, 2.0 * self.barrier_thickness, 0.5 * self.width)
        self._default_padding_x = default_padding
        self._default_padding_y = default_padding

    @property
    def local_bounds(self) -> Bounds2D:
        return self._local_bounds

    @property
    def default_padding_x(self) -> float:
        return self._default_padding_x

    @property
    def default_padding_y(self) -> float:
        return self._default_padding_y

    @property
    def characteristic_width(self) -> float:
        return self.width

    @property
    def inner_half_width(self) -> float:
        return self._inner_half_width

    @property
    def open_end_y(self) -> float:
        return self._open_end_y

    def _build_local_segments(self) -> list[SegmentPrimitive]:
        global_offset_x = -self.width / 2.0
        global_offset_y = self.width / 2.0
        r = self.width / 2.0 + self.barrier_thickness
        xs = np.linspace(-r, r, self.arc_samples)
        ys = -semi_circle(r, xs)

        segments = []
        for x, y in zip(xs, ys):
            x1 = x + dx(r, x, 5.0) + r - self.barrier_thickness
            y1 = y - dy(r, x, 5.0)
            x2 = x - dx(r, x, 5.0) + r - self.barrier_thickness
            y2 = y + dy(r, x, 5.0)
            segments.append(
                SegmentPrimitive(
                    p1=(global_offset_x + x1, global_offset_y + y1),
                    p2=(global_offset_x + x2, global_offset_y + y2),
                    thickness=self.barrier_thickness,
                )
            )

        segments.extend(
            [
                SegmentPrimitive(
                    p1=(-self.width / 2.0 - self.barrier_thickness, self.width / 2.0),
                    p2=(-self.width / 2.0 - self.barrier_thickness, self.width / 2.0 + self.trench_length),
                    thickness=self.barrier_thickness,
                ),
                SegmentPrimitive(
                    p1=(self.width / 2.0 + self.barrier_thickness, self.width / 2.0),
                    p2=(self.width / 2.0 + self.barrier_thickness, self.width / 2.0 + self.trench_length),
                    thickness=self.barrier_thickness,
                ),
            ]
        )
        return segments

    @staticmethod
    def _compute_local_bounds(segments: list[SegmentPrimitive]) -> Bounds2D:
        min_x = min(min(segment.p1[0], segment.p2[0]) - segment.thickness for segment in segments)
        min_y = min(min(segment.p1[1], segment.p2[1]) - segment.thickness for segment in segments)
        max_x = max(max(segment.p1[0], segment.p2[0]) + segment.thickness for segment in segments)
        max_y = max(max(segment.p1[1], segment.p2[1]) + segment.thickness for segment in segments)
        return Bounds2D(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

    def build(self, space: pymunk.Space, layout: GeometryLayout) -> None:
        for segment in self.preview_primitives(layout):
            _add_world_segment(space, segment)

    def preview_primitives(self, layout: GeometryLayout) -> list[SegmentPrimitive]:
        dx_world, dy_world = layout.world_offset
        return [segment.translated(dx_world, dy_world) for segment in self._local_segments]

    def seed_cell_local_position(self, segment_radius: float) -> tuple[float, float]:
        return (0.0, self.width / 2.0 + segment_radius * 3.0)

    def positions_within_bounds(
        self,
        positions,
        radii,
        layout: GeometryLayout,
        *,
        enforce_open_end_cap: bool,
    ) -> bool:
        local_positions = layout.to_local_points(positions)
        for position, radius in zip(local_positions, radii):
            radius = float(radius)
            x = float(position[0])
            y = float(position[1])
            if x < (-self.inner_half_width + radius):
                return False
            if x > (self.inner_half_width - radius):
                return False
            if y < (0.25 * radius):
                return False
            if enforce_open_end_cap and y > (self.open_end_y - 0.25 * radius):
                return False
        return True

    def cell_out_of_bounds(self, positions, radii, layout: GeometryLayout) -> bool:
        local_positions = layout.to_local_points(positions)
        for position, radius in zip(local_positions, radii):
            radius = float(radius)
            y = float(position[1])
            if y < (-0.25 * radius):
                return True
            if y > (self.open_end_y + 0.25 * radius):
                return True
        return False

    def project_body_inside_bounds(self, body, radius: float, layout: GeometryLayout) -> tuple[bool, bool, bool]:
        local_position = layout.to_local_point((float(body.position[0]), float(body.position[1])))
        min_x = -self.inner_half_width + radius
        max_x = self.inner_half_width - radius
        min_y = 0.25 * radius

        clamped_x = min(max(local_position[0], min_x), max_x)
        clamped_y = max(local_position[1], min_y)
        changed_x = clamped_x != local_position[0]
        changed_y = clamped_y != local_position[1]
        projected = changed_x or changed_y
        if projected:
            world_x, world_y = layout.to_world_point((clamped_x, clamped_y))
            vec_type = body.position.__class__
            body.position = vec_type(world_x, world_y)
            if hasattr(body, "velocity"):
                velocity = getattr(body, "velocity")
                velocity = vec_type(float(velocity[0]), float(velocity[1]))
                if changed_x:
                    velocity = vec_type(0.0, float(velocity[1]))
                if changed_y:
                    velocity = vec_type(float(velocity[0]), 0.0)
                body.velocity = velocity
        return projected, changed_x, changed_y


def trench_creator(width, trench_length, global_xy, space, barrier_thickness=10):
    """Backward-compatible trench builder using the new geometry definition."""
    geometry = TrenchGeometrySpec(width=width, trench_length=trench_length, barrier_thickness=barrier_thickness)
    for segment in geometry._local_segments:
        _add_world_segment(space, segment.translated(float(global_xy[0]), float(global_xy[1])))


def box_creator(width, height, global_xy, space, barrier_thickness=10, fillet_radius=20, fillet_segments=10):
    """
    Creates an open-ended box made of static walls and adds it to a Pymunk space.
    The box features adjustable fillets on the two corners of its closed end.

    Args:
        width (float): The inner width of the box.
        height (float): The inner height of the box walls.
        global_xy (tuple): The center of the box's closed side (bottom inner edge).
        space (pymunk.Space): The Pymunk space to add the box to.
        barrier_thickness (float): The thickness of the walls.
        fillet_radius (float): The radius of the corner fillets. A value of 0 creates sharp corners.
        fillet_segments (int): The number of segments used to approximate each corner arc.
    """
    fillet_radius = min(abs(fillet_radius), width / 2, height)
    if fillet_radius > 0:
        fillet_segments = max(1, fillet_segments)

    segments_to_add = []

    p1 = (-width / 2, height)
    p2 = (-width / 2, fillet_radius)
    segments_to_add.append(segment_creator(p1, p2, global_xy, barrier_thickness))

    p1 = (width / 2, height)
    p2 = (width / 2, fillet_radius)
    segments_to_add.append(segment_creator(p1, p2, global_xy, barrier_thickness))

    p1 = (-width / 2 + fillet_radius, 0)
    p2 = (width / 2 - fillet_radius, 0)
    segments_to_add.append(segment_creator(p1, p2, global_xy, barrier_thickness))

    if fillet_radius > 0 and fillet_segments > 0:
        cx_left = -width / 2 + fillet_radius
        cy_left = fillet_radius
        angles_left = np.linspace(np.pi, 1.5 * np.pi, fillet_segments + 1)
        points_left = [(cx_left + fillet_radius * np.cos(a), cy_left + fillet_radius * np.sin(a)) for a in angles_left]
        for i in range(fillet_segments):
            segments_to_add.append(segment_creator(points_left[i], points_left[i + 1], global_xy, barrier_thickness))

        cx_right = width / 2 - fillet_radius
        cy_right = fillet_radius
        angles_right = np.linspace(1.5 * np.pi, 2 * np.pi, fillet_segments + 1)
        points_right = [(cx_right + fillet_radius * np.cos(a), cy_right + fillet_radius * np.sin(a)) for a in angles_right]
        for i in range(fillet_segments):
            segments_to_add.append(segment_creator(points_right[i], points_right[i + 1], global_xy, barrier_thickness))

    for body, shape in segments_to_add:
        space.add(body, shape)
