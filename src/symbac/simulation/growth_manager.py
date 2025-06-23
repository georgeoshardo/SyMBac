import numpy as np
from symbac.simulation.cell import Cell

# Follow the ECS paradigm and have a separate growth manager class that handles the growth of cells and is stateless
# Can consider making this manager take a nutrient profile to adjust per cell growth rate. For now constant
class GrowthManager:

    def grow(self, cell: Cell, dt: float):
        """
        Handles the growth of the cell by elongating the joints at the head and tail.
        When a joint is stretched beyond a threshold, it triggers the insertion of a new segment.
        """
        if not cell.is_dividing and len(cell.PhysicsRepresentation.segments) >= cell._max_length:
            return

        if len(cell.PhysicsRepresentation.segments) < 2:
            return

        # Distribute growth evenly to both ends of the cell
        added_length = cell.adjusted_growth_rate * dt * np.random.uniform(0, 4)
        half_growth = added_length / 2

        cell.PhysicsRepresentation.growth_accumulator_head += half_growth
        cell.PhysicsRepresentation.growth_accumulator_tail += half_growth

        # Stretch the head joint by adjusting the anchor on the first segment.
        # This pushes the head segment outwards.
        first_pivot_joint = cell.PhysicsRepresentation.pivot_joints[0]
        first_pivot_joint.anchor_a = (
            cell.config.JOINT_DISTANCE / 2 + cell.PhysicsRepresentation.growth_accumulator_head,
            0
        )

        # Stretch the tail joint by adjusting the anchor on the last segment.
        # This pushes the tail segment outwards.
        last_pivot_joint = cell.PhysicsRepresentation.pivot_joints[-1]
        last_pivot_joint.anchor_b = (
            -cell.config.JOINT_DISTANCE / 2 - cell.PhysicsRepresentation.growth_accumulator_tail,
            0,
        )

        # If the head has grown enough, insert a new segment.
        if cell.PhysicsRepresentation.growth_accumulator_head >= cell.config.GROWTH_THRESHOLD:
            cell.PhysicsRepresentation.add_head_segment()
            cell.PhysicsRepresentation.growth_accumulator_head = 0.0

        # If the tail has grown enough, insert a new segment.
        if cell.PhysicsRepresentation.growth_accumulator_tail >= cell.config.GROWTH_THRESHOLD:
            cell.PhysicsRepresentation.add_tail_segment()
            cell.PhysicsRepresentation.growth_accumulator_tail = 0.0

        cell._update_colors()

