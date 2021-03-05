import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions

window = pyglet.window.Window(1280, 720, "Pymunk Tester", resizable=False)
options = DrawOptions()

space = pymunk.Space()
space.gravity = 0, -100


mass = 1
radius = 20
circle_moment = pymunk.moment_for_circle(mass, 0, radius)
circle_body = pymunk.Body(mass, circle_moment)
circle_body.position = 430, 700
circle_shape = pymunk.Circle(circle_body, radius)
circle_shape.friction=10
space.add(circle_body, circle_shape)

mass = 1
radius = 20
circle_moment = pymunk.moment_for_circle(mass, 0, radius)
circle_body = pymunk.Body(mass, circle_moment)
circle_body.position = 430, 500
circle_shape = pymunk.Circle(circle_body, radius)
circle_shape.friction=10
space.add(circle_body, circle_shape)

mass = 1
radius = 20
circle_moment = pymunk.moment_for_circle(mass, 0, radius)
circle_body = pymunk.Body(mass, circle_moment)
circle_body.position = 430, 600
circle_shape = pymunk.Circle(circle_body, radius)
circle_shape.friction=10
space.add(circle_body, circle_shape)



def segment_creator(local_xy1, local_xy2,global_xy):
    segment_body = pymunk.Body(mass=2, moment=3,body_type=pymunk.Body.STATIC)
    segment_shape = pymunk.Segment(segment_body, local_xy1,local_xy2,0)
    segment_body.position = global_xy
    segment_shape.friction = 10
    return segment_body, segment_shape


def trench_creator(size,global_xy, space):
    trench_coords = []
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

    trench_length = 300
    left_wall = segment_creator((0,0),(0,trench_length),global_xy)
    right_wall = segment_creator((size,0),(size,trench_length),(global_xy[0]+size/2, global_xy[1]))
    walls = [left_wall, right_wall]
    for z in walls:
        for s in z:
            space.add(s)


trench_creator(30,(400,100), space)



@window.event
def on_draw():
    window.clear()
    space.debug_draw(options)

def update(dt):
    space.step(1/30)

if __name__ == "__main__":
    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()