"""Adapter that snapshots a SimCell for the drawing and lineage pipelines."""
import numpy as np
import pymunk
from SyMBac import cell_geometry


class _FakeShape:
    """Mimics pymunk.Poly so code calling shape.get_vertices() still works."""

    def __init__(self, body, length, width, resolution=60):
        self.body = body
        self._length = length
        self._width = width
        self._resolution = resolution

    def get_vertices(self):
        return cell_geometry.get_vertices(
            self._length, self._width, 0, self._resolution
        )


class CellSnapshot:
    """Immutable snapshot of a SimCell at one timepoint.

    Stores both the raw segment chain geometry (for segment-based OPL
    drawing) and legacy scalar properties expected by the old drawing
    pipeline, lineage code, and any external consumer.

    Parameters
    ----------
    simcell : SyMBac.physics.simcell.SimCell
        The live SimCell to snapshot.
    t : int
        Frame index.
    mother_mask_label : int or None
        mask_label of the mother cell (set by division hook).
    generation : int
        Generation counter.
    just_divided : bool
        Whether this cell just divided this frame.
    """

    __slots__ = (
        # Segment data
        'segment_positions',
        'segment_radii',
        # Legacy scalars
        'length',
        'width',
        'angle',
        'position',
        'pinching_sep',
        'mask_label',
        'ID',
        'mother_mask_label',
        'generation',
        'N_divisions',
        't',
        'dead',
        'lysis_p',
        'just_divided',
        # Fake pymunk objects
        'body',
        'shape',
    )

    def __init__(
        self,
        simcell,
        t=0,
        mother_mask_label=None,
        generation=0,
        just_divided=False,
        dead=False,
        lysis_p=0.0,
    ):
        pr = simcell.physics_representation

        # --- A. Segment chain data ---
        self.segment_positions = np.array(
            [tuple(s.position) for s in pr.segments], dtype=np.float64
        )
        self.segment_radii = np.array(
            [s.radius for s in pr.segments], dtype=np.float64
        )

        # --- B. Legacy scalar properties ---
        self.length = pr.get_continuous_length()
        self.width = 2.0 * simcell.config.SEGMENT_RADIUS

        # Angle: direction from first to last segment
        if len(pr.segments) >= 2:
            first = np.array(tuple(pr.segments[0].position))
            last = np.array(tuple(pr.segments[-1].position))
            diff = last - first
            self.angle = float(np.arctan2(diff[1], diff[0]))
        else:
            self.angle = 0.0

        # Position: mean of segment positions
        if len(self.segment_positions) > 0:
            self.position = pymunk.Vec2d(*np.mean(self.segment_positions, axis=0))
        else:
            self.position = pymunk.Vec2d(0, 0)

        # Pinching: derive from septum_progress
        self.pinching_sep = simcell.septum_progress * self.width if simcell.is_dividing else 0.0

        self.mask_label = simcell.group_id
        self.ID = simcell.group_id
        self.mother_mask_label = mother_mask_label
        self.generation = generation
        self.N_divisions = simcell.num_divisions
        self.t = t
        self.dead = dead
        self.lysis_p = lysis_p
        self.just_divided = just_divided

        # --- C. Fake pymunk body + shape ---
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = self.position
        body.angle = self.angle
        self.body = body
        self.shape = _FakeShape(body, self.length, self.width)

    def to_segment_dict(self):
        """Return dict suitable for draw_scene_from_segments()."""
        return {
            'positions': self.segment_positions,
            'radii': self.segment_radii,
            'mask_label': self.mask_label,
            'cell_id': self.ID,
        }
