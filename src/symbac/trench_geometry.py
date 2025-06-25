import numpy as np
import pandas as pd
import pymunk


def segment_creator(local_xy1, local_xy2, global_xy, thickness):
    segment_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment_shape = pymunk.Segment(segment_body, local_xy1, local_xy2, thickness)
    segment_body.position = global_xy
    segment_shape.friction = 0
    return segment_body, segment_shape


def semi_circle(r, x):
    return np.sqrt(r**2 - x**2)


def semi_circle_grad(r, x):
    return -x / np.sqrt(r**2 - x**2)


def y_int(r, x):
    return semi_circle_grad(r, x) * (-x) + semi_circle(r, x)


def x_int(r, x):
    return -semi_circle(r, x) / semi_circle_grad(r, x) + x


def dx(r, x, distance):
    return distance * np.cos(np.arctan(semi_circle_grad(r, x)))


def dy(r, x, distance):
    return distance * np.sin(np.arctan(semi_circle_grad(r, x)))


def trench_creator(width, trench_length, global_xy, space, barrier_thickness=10):
    global_xy = global_xy[0] - width/2, global_xy[1] + width/2
    r = width / 2 + barrier_thickness
    xs = np.linspace(-r, r, 50)
    ys = -semi_circle(r, xs)

    segments = []
    for x, y in zip(xs, ys):
        x1 = x + dx(r, x, 5) + r - barrier_thickness
        y1 = y - dy(r, x, 5)
        x2 = x - dx(r, x, 5) + r - barrier_thickness
        y2 = y + dy(r, x, 5)

        segment = segment_creator((x1, y1), (x2, y2), global_xy, barrier_thickness)
        segments.append(segment)

    for z in segments:
        for s in z:
            space.add(s)

    left_barrier = segment_creator(
        (0, 0),
        (0, trench_length),
        (global_xy[0] - barrier_thickness, global_xy[1]),
        barrier_thickness,
    )
    right_barrier = segment_creator(
        (0, 0),
        (0, trench_length),
        (global_xy[0] + width + barrier_thickness, global_xy[1]),
        barrier_thickness,
    )
    walls = [left_barrier, right_barrier]
    for z in walls:
        for s in z:
            space.add(s)


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
    # --- Input Validation and Setup ---

    # Cap the fillet radius to prevent geometric errors where fillets would overlap.
    fillet_radius = min(abs(fillet_radius), width / 2, height)

    # Ensure there's at least one segment for the arc if a radius is provided.
    if fillet_radius > 0:
        fillet_segments = max(1, fillet_segments)

    segments_to_add = []

    # --- Wall Geometry Calculation ---
    # All segment endpoints are calculated in a local coordinate system where (0,0)
    # is the center of the bottom inner edge (global_xy). Each segment's body
    # is then placed at global_xy.

    # 1. Straight Left Wall
    p1 = (-width / 2, height)
    p2 = (-width / 2, fillet_radius)
    segments_to_add.append(segment_creator(p1, p2, global_xy, barrier_thickness))

    # 2. Straight Right Wall
    p1 = (width / 2, height)
    p2 = (width / 2, fillet_radius)
    segments_to_add.append(segment_creator(p1, p2, global_xy, barrier_thickness))

    # 3. Straight Bottom Wall
    p1 = (-width / 2 + fillet_radius, 0)
    p2 = (width / 2 - fillet_radius, 0)
    segments_to_add.append(segment_creator(p1, p2, global_xy, barrier_thickness))

    # --- Fillet Generation ---
    if fillet_radius > 0 and fillet_segments > 0:
        # 4. Bottom-Left Fillet
        # The center of the arc for the left fillet
        cx_left = -width / 2 + fillet_radius
        cy_left = fillet_radius

        # Generate points along the arc from 180 to 270 degrees
        angles_left = np.linspace(np.pi, 1.5 * np.pi, fillet_segments + 1)
        points_left = [(cx_left + fillet_radius * np.cos(a), cy_left + fillet_radius * np.sin(a)) for a in angles_left]

        # Create small segments between the points to form the arc
        for i in range(fillet_segments):
            segments_to_add.append(segment_creator(points_left[i], points_left[i + 1], global_xy, barrier_thickness))

        # 5. Bottom-Right Fillet
        # The center of the arc for the right fillet
        cx_right = width / 2 - fillet_radius
        cy_right = fillet_radius

        # Generate points along the arc from 270 to 360 degrees
        angles_right = np.linspace(1.5 * np.pi, 2 * np.pi, fillet_segments + 1)
        points_right = [(cx_right + fillet_radius * np.cos(a), cy_right + fillet_radius * np.sin(a)) for a in
                        angles_right]

        for i in range(fillet_segments):
            segments_to_add.append(segment_creator(points_right[i], points_right[i + 1], global_xy, barrier_thickness))

    # Add all created shapes and bodies to the Pymunk space
    for body, shape in segments_to_add:
        space.add(body, shape)

def get_trench_segments(space):
    """
    A function which extracts the rigid body trench objects from the pymunk space object. Space object should be passed
    from the return value of the run_simulation() function

    Returns
    -------
    List of trench segment properties, later used to draw the trench.
    """
    trench_shapes = []
    for shape, body in zip(space.shapes, space.bodies):
        if body.body_type == 2:
            trench_shapes.append(shape)

    trench_segment_props = []
    for x in trench_shapes:
        trench_segment_props.append([x.bb, x.area, x.a, x.b])

    trench_segment_props = pd.DataFrame(trench_segment_props)
    trench_segment_props.columns = ["bb", "area", "a", "b"]
    main_segments = trench_segment_props.sort_values("area", ascending=False).iloc[0:2]
    return main_segments
