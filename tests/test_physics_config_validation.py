import pytest
import pymunk

from SyMBac.physics.config import CellConfig, PhysicsConfig
from SyMBac.physics.segments import CellSegment
from SyMBac.physics.simulator import Simulator


def test_physics_config_dt_is_constructible_and_overridable():
    cfg = PhysicsConfig(DT=0.01)
    assert cfg.DT == pytest.approx(0.01)


def test_physics_config_collision_slop_zero_is_applied():
    sim = Simulator(
        physics_config=PhysicsConfig(COLLISION_SLOP=0.0),
        initial_cell_config=CellConfig(),
    )
    assert sim.space.collision_slop == pytest.approx(0.0)


def test_physics_config_rejects_negative_values():
    with pytest.raises(ValueError, match="DT"):
        PhysicsConfig(DT=0.0)
    with pytest.raises(ValueError, match="COLLISION_SLOP"):
        PhysicsConfig(COLLISION_SLOP=-0.1)


def test_cell_config_defaults_are_constructible():
    cfg = CellConfig()
    assert cfg.ROTARY_LIMIT_JOINT is True
    assert cfg.MAX_BEND_ANGLE is not None
    assert cfg.STIFFNESS is not None


def test_cell_config_rejects_invalid_geometry_values():
    with pytest.raises(ValueError, match="GRANULARITY"):
        CellConfig(GRANULARITY=0)
    with pytest.raises(ValueError, match="SEGMENT_RADIUS"):
        CellConfig(SEGMENT_RADIUS=0.0)
    with pytest.raises(ValueError, match="SEGMENT_MASS"):
        CellConfig(SEGMENT_MASS=0.0)
    with pytest.raises(ValueError, match="SEED_CELL_SEGMENTS"):
        CellConfig(SEED_CELL_SEGMENTS=1)


def test_segment_radius_updates_body_moment():
    segment = CellSegment(
        config=CellConfig(SEGMENT_RADIUS=10.0),
        group_id=1,
        position=pymunk.Vec2d(0, 0),
    )

    original_moment = segment.body.moment
    segment.set_radius(30.0)

    assert segment.radius == pytest.approx(30.0)
    assert segment.body.moment != pytest.approx(original_moment)
    assert segment.body.moment == pytest.approx(
        pymunk.moment_for_circle(segment.body.mass, 0, 30.0)
    )
