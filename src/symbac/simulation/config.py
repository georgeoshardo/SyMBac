from dataclasses import dataclass, field
import numpy as np

@dataclass(slots=True, frozen=True)
class CellConfig:
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

    DAMPED_ROTARY_SPRING: bool = False
    ROTARY_SPRING_STIFFNESS: float | None = None
    ROTARY_SPRING_DAMPING: float | None = None

    ROTARY_LIMIT_JOINT: bool = True  # Whether to use rotary limit joints
    MAX_BEND_ANGLE: float | None = None  # 0.01 normally, 0.05 also good for E. coli in MM
    STIFFNESS: float | None = None  # Stiffness for limit joints, can be np.inf for max stiffness
    JOINT_DISTANCE: float = field(init=False)
    GROWTH_THRESHOLD: float = field(init=False)

    def __post_init__(self):

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



class SimViewerConfig:
    """ General simulation settings. """
    SCREEN_WIDTH: int = 1200
    SCREEN_HEIGHT: int = 800
    FPS: int = 1
    SIMULATION_SPEED_MULTIPLIER: int = 10
    BACKGROUND_COLOR: tuple[int, int, int] = (20, 30, 40)
    FONT_SIZE: int = 36

class PhysicsConfig:
    ITERATIONS: int = 60
    DAMPING: float = 0.5
    GRAVITY: tuple[float, float] = (0.0, 0.0)
    THREADED: bool = False # Use pymunk.Space(threaded=True) for multithreading but non-deterministic results, even with random seed set
    THREADS: int = 1