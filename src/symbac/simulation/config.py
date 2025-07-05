from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pymunk.vec2d import Vec2d


@dataclass(slots=True, frozen=True)
class CellConfig:
    """Parameters to configure a cell."""
    GRANULARITY: int = 8  # Number of segments per cell radius
    SEGMENT_RADIUS: float = 15.0
    SEGMENT_MASS: float = 1.0
    GROWTH_RATE: float = 5.0
    MIN_LENGTH_AFTER_DIVISION: int = 10
    MAX_LENGTH_VARIATION: float = 0.2
    BASE_MAX_LENGTH: int = 40
    SEED_CELL_SEGMENTS: int = 15
    PIVOT_JOINT_STIFFNESS: float = np.inf  # Stiffness for pivot joints
    NOISE_STRENGTH: float = 0.05
    START_POS: Vec2d = (0.0, 0.0)  # Starting position of the cell
    START_ANGLE: float = 0.0 #in RADIANS!!
    #Septum configuration
    SEPTUM_DURATION: float = 1.5  # Duration of septum formation in seconds

    DAMPED_ROTARY_SPRING: bool = False
    ROTARY_SPRING_STIFFNESS: float | None = None
    ROTARY_SPRING_DAMPING: float | None = None

    ROTARY_LIMIT_JOINT: bool = True  # Whether to use rotary limit joints
    MAX_BEND_ANGLE: float | None = None  # 0.01 normally, 0.05 also good for E. coli in MM
    STIFFNESS: float | None = None  # Stiffness for limit joints, can be np.inf for max stiffness

    #Parameters to be set post init
    JOINT_DISTANCE: float = field(init=False)
    GROWTH_THRESHOLD: float = field(init=False)
    MIN_SEPTUM_RADIUS: float = field(init=False)
    NUM_SEPTUM_SEGMENTS: int = field(init=False)

    def __post_init__(self):

        if isinstance(self.START_POS, tuple):
            object.__setattr__(self, "START_POS", Vec2d(*self.START_POS))

        if not self.DAMPED_ROTARY_SPRING:
            if self.ROTARY_SPRING_STIFFNESS is not None or self.ROTARY_SPRING_DAMPING is not None:
                raise ValueError(
                    "Cannot set ROTARY_SPRING_STIFFNESS or ROTARY_SPRING_DAMPING "
                    "when ROTARY_SPRING is False."
                )
        else:
            if self.ROTARY_SPRING_STIFFNESS is None:
                raise ValueError(
                    "DAMPED_ROTARY_SPRING=True, but ROTARY_SPRING_STIFFNESS was not provided."
                )
            if self.ROTARY_SPRING_DAMPING is None:
                raise ValueError(
                    "DAMPED_ROTARY_SPRING=True, but ROTARY_SPRING_DAMPING was not provided."
                )

        if not self.ROTARY_LIMIT_JOINT:
            if self.MAX_BEND_ANGLE is not None or self.STIFFNESS is not None:
                raise ValueError(
                    "Cannot set MAX_BEND_ANGLE or STIFFNESS when ROTARY_LIMIT_JOINT is False."
                )
        else:
            if self.MAX_BEND_ANGLE is None:
                raise ValueError(
                    "ROTARY_LIMIT_JOINT=False, but MAX_BEND_ANGLE was not provided."
                )
            if self.STIFFNESS is None:
                raise ValueError(
                    "ROTARY_LIMIT_JOINT=False, but STIFFNESS was not provided."
                )

        # Use object.__setattr__ because the class is frozen
        object.__setattr__(self, "JOINT_DISTANCE", self.SEGMENT_RADIUS / self.GRANULARITY)
        object.__setattr__(self, "GROWTH_THRESHOLD", self.SEGMENT_RADIUS / self.GRANULARITY)
        object.__setattr__(self, "MIN_SEPTUM_RADIUS", self.SEGMENT_RADIUS * 0.1)
        object.__setattr__(self, "NUM_SEPTUM_SEGMENTS", self.GRANULARITY)

@dataclass(slots=True, frozen=True)
class SimViewerConfig:
    """ General simulation settings. """
    SCREEN_WIDTH: int = 1200
    SCREEN_HEIGHT: int = 800
    FPS: int = 1
    SIM_STEPS_PER_DRAW: int = 10
    BACKGROUND_COLOR: tuple[int, int, int] = (20, 30, 40)
    FONT_SIZE: int = 36

@dataclass(slots=True, frozen=True)
class PhysicsConfig:
    """ Physics-specific settings. """
    ITERATIONS: int = 120 # Number of iterations for the physics simulation, long floppy cells might need to go higher than 60
    DAMPING: float = 0.5
    GRAVITY: tuple[float, float] = (0.0, 0.0)
    THREADED: bool = False # Use pymunk.Space(threaded=True) for multithreading but non-deterministic results, even with random seed set
    THREADS: int = 1
    DT = 1.0 / 60.0  # Time step for the physics simulation
    COLLISION_SLOP: Optional[float] = False # Amount of overlap between shapes that is allowed. To improve stability, set this as high as you can without noticeable overlapping. It defaults to 0.1.

    def __post_init__(self):
        if self.THREADS > 2:
            raise ValueError("THREADS cannot be greater than 2.")
        if not self.THREADED and self.THREADS != 1:
            raise ValueError("If THREADED is False, THREADS must be 1.")
        if self.THREADED and self.THREADS != 2:
            raise ValueError("If THREADED is True, THREADS must be 2.")