import pytest

from SyMBac.physics.config import CellConfig, PhysicsConfig
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


@pytest.mark.parametrize(
    "config_type,kwargs,error",
    [
        (CellConfig, {"GRANULARITY": 0}, "GRANULARITY"),
        (CellConfig, {"GRANULARITY": 1.5}, "GRANULARITY"),
        (CellConfig, {"SEPTUM_DURATION": 0.0}, "SEPTUM_DURATION"),
        (CellConfig, {"SEPTUM_DURATION": float("inf")}, "SEPTUM_DURATION"),
        (CellConfig, {"MIN_LENGTH_AFTER_DIVISION": 1}, "MIN_LENGTH_AFTER_DIVISION"),
        (CellConfig, {"MIN_LENGTH_AFTER_DIVISION": 2.5}, "MIN_LENGTH_AFTER_DIVISION"),
        (CellConfig, {"SEED_CELL_SEGMENTS": 1}, "SEED_CELL_SEGMENTS"),
        (CellConfig, {"SEED_CELL_SEGMENTS": 2.5}, "SEED_CELL_SEGMENTS"),
        (PhysicsConfig, {"DT": float("inf")}, "DT"),
        (PhysicsConfig, {"ITERATIONS": 0}, "ITERATIONS"),
        (PhysicsConfig, {"ITERATIONS": 1.5}, "ITERATIONS"),
        (PhysicsConfig, {"DAMPING": -0.1}, "DAMPING"),
        (PhysicsConfig, {"DAMPING": 1.1}, "DAMPING"),
        (PhysicsConfig, {"DAMPING": float("nan")}, "DAMPING"),
    ],
)
def test_configs_reject_invalid_core_values(config_type, kwargs, error):
    with pytest.raises(ValueError, match=error):
        config_type(**kwargs)
