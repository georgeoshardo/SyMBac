import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions
from SYMPTOMM import cell_geometry

window = pyglet.window.Window(1280, 720, "Pymunk Tester", resizable=False)
options = DrawOptions()
@window.event
def on_draw():
    window.clear()
    space.debug_draw(options)

space = pymunk.Space()
space.gravity = 0, 0

cell_vertices = cell_geometry.get_vertices(100,50,20)
cell_shape = pymunk.Poly(None, cell_vertices)
cell_moment = 1
cell_mass = 1
cell_body = pymunk.Body(cell_mass,cell_moment)
cell_shape.body = cell_body
cell_body.position = 200,200

cell_vertices2 = cell_geometry.get_vertices(100,60,20)
cell_shape2 = pymunk.Poly(None, cell_vertices2)
cell_moment2 = 100
cell_mass2 = 100
cell_body2 = pymunk.Body(cell_mass,cell_moment)
cell_shape2.body = cell_body2
cell_body2.position = 270,200

space.add(cell_body, cell_shape,cell_body2, cell_shape2)


def update(dt):
    space.step(dt)





if __name__ == "__main__":
    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()


