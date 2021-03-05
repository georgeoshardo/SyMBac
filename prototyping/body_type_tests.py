import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions

window = pyglet.window.Window(1280, 720, "Pymunk Tester", resizable=False)
options = DrawOptions()

space = pymunk.Space()
space.gravity = 0, -100

poly = pymunk.Poly.create_box(None, size=(150,150))
moment = pymunk.moment_for_poly(10, poly.get_vertices())

body = pymunk.Body(1,moment, pymunk.Body.DYNAMIC)
poly.body = body
body.position = 640, 600

space.add(body,poly)

@window.event
def on_draw():
    window.clear()
    space.debug_draw(options)

def update(dt):
    space.step(dt)

if __name__ == "__main__":
    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()