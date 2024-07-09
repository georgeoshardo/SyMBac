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


def trench_creator(size, trench_length, global_xy, space):
    r = size / 2
    xs = np.linspace(-r, r, 50)
    ys = -semi_circle(r, xs)

    segments = []
    for x, y in zip(xs, ys):
        x1 = x + dx(r, x, 5) + r
        y1 = y - dy(r, x, 5)
        x2 = x - dx(r, x, 5) + r
        y2 = y + dy(r, x, 5)

        segment = segment_creator((x1, y1), (x2, y2), global_xy, 1)
        segments.append(segment)

    # size = int(np.ceil(size/1.5))
    # segments = []
    # for x in range(size):
    #     segment = segment_creator((x,0),(0,size-x),global_xy,1)
    #     segments.append(segment)

    # for x in range(size):
    #     segment = segment_creator((size-x,0),(size,size-x),(global_xy[0]+size/2, global_xy[1]),1)
    #     segments.append(segment)
    for z in segments:
        for s in z:
            space.add(s)

    left_wall = segment_creator((0, 0), (0, trench_length), global_xy, 1)
    right_wall = segment_creator((0, 0), (0, trench_length), (global_xy[0] + size, global_xy[1]), 1)
    barrier_thickness = 5
    left_barrier = segment_creator(
        (0, 0),
        (0, trench_length),
        (global_xy[0] - barrier_thickness, global_xy[1]),
        barrier_thickness,
    )
    right_barrier = segment_creator(
        (0, 0),
        (0, trench_length),
        (global_xy[0] + size + barrier_thickness, global_xy[1]),
        barrier_thickness,
    )
    walls = [left_wall, right_wall, left_barrier, right_barrier]
    for z in walls:
        for s in z:
            space.add(s)


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
