import numpy as np
from symbac.simulation.simcell import SimCell

# Follow the ECS paradigm and have a separate growth manager class that handles the growth of cells and is stateless
# Can consider making this manager take a nutrient profile to adjust per cell growth rate. For now constant
class GrowthManager:
    """Manages the growth of cells by elongating them and triggering new segment insertion.

    This manager follows the Entity-Component-System (ECS) paradigm and is stateless,
    focusing solely on the growth logic.
    """
    @staticmethod
    def grow(cell: SimCell, dt: float):
        """Handles the growth of the cell by elongating the joints at the head and tail.

        When a joint is stretched beyond a configurable threshold, it triggers
        the insertion of a new segment at that end of the cell. Growth is
        distributed evenly to both ends.

        Args:
            cell: The `SimCell` object to be grown.
            dt: The time step for the current simulation frame.
        """
        if not cell.is_dividing and cell.length >= cell.max_length: #LENGTH_FIX # TODO: is this actually necessary?
            return

        if cell.physics_representation.num_segments < 2:
            return

        # Distribute growth evenly to both ends of the cell
        added_length = cell.adjusted_growth_rate * dt * np.random.uniform(0, 4)
        half_growth = added_length / 2

        cell.physics_representation.growth_accumulator_head += half_growth
        cell.physics_representation.growth_accumulator_tail += half_growth

        # Stretch the head joint by adjusting the anchor on the first segment.
        # This pushes the head segment outwards.
        first_pivot_joint = cell.physics_representation.pivot_joints[0]
        first_pivot_joint.anchor_a = (
            cell.config.JOINT_DISTANCE / 2 + cell.physics_representation.growth_accumulator_head,
            0
        )

        # Stretch the tail joint by adjusting the anchor on the last segment.
        # This pushes the tail segment outwards.
        last_pivot_joint = cell.physics_representation.pivot_joints[-1]
        last_pivot_joint.anchor_b = (
            -cell.config.JOINT_DISTANCE / 2 - cell.physics_representation.growth_accumulator_tail,
            0,
        )

        # If the head has grown enough, insert a new segment.
        if cell.physics_representation.growth_accumulator_head >= cell.config.GROWTH_THRESHOLD:
            cell.physics_representation.add_head_segment()
            cell.physics_representation.growth_accumulator_head = 0.0

        # If the tail has grown enough, insert a new segment.
        if cell.physics_representation.growth_accumulator_tail >= cell.config.GROWTH_THRESHOLD:
            cell.physics_representation.add_tail_segment()
            cell.physics_representation.growth_accumulator_tail = 0.0
