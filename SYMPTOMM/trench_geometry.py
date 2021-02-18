import pymunk

def segment_creator(local_xy1, local_xy2,global_xy):
    segment_body = pymunk.Body(mass=2, moment=3,body_type=pymunk.Body.STATIC)
    segment_shape = pymunk.Segment(segment_body, local_xy1,local_xy2,20)
    segment_body.position = global_xy
    segment_shape.friction = 1000
    return segment_body, segment_shape


def trench_creator(size,trench_length, global_xy, space):
    segments = []
    for x in range(size):
        segment = segment_creator((x,0),(0,size-x),global_xy)
        segments.append(segment)

    for x in range(size):
        segment = segment_creator((size-x,0),(size,size-x),(global_xy[0]+size/2, global_xy[1]))
        segments.append(segment)
    for z in segments:
        for s in z:
            space.add(s)

    left_wall = segment_creator((0,0),(0,trench_length),global_xy)
    right_wall = segment_creator((size,0),(size,trench_length),(global_xy[0]+size/2, global_xy[1]))
    walls = [left_wall, right_wall]
    for z in walls:
        for s in z:
            space.add(s)

    