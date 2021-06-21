#%%
import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions
import sys
sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2')
from SYMPTOMM import cell_geometry
import matplotlib.pyplot as plt
import numpy as np

window = pyglet.window.Window(1280, 720, "Pymunk Tester", resizable=False)
options = DrawOptions()

space = pymunk.Space()
space.gravity = 0.0,0.0

#%%
def create_cell(length, width, resolution, angle, position):
    cell_vertices = cell_geometry.get_vertices(length,width,angle, resolution)
    cell_shape = pymunk.Poly(None, cell_vertices)
    cell_moment = 1
    cell_mass = 0.00001
    cell_body = pymunk.Body(cell_mass,cell_moment)
    cell_shape.body = cell_body
    cell_body.position = position
    cell_body.angle = angle
    cell_shape.friction=1000
    return cell_body, cell_shape, length, width, resolution, angle




@window.event
def on_draw():
    window.clear()
    space.debug_draw(options)

a = 0
def update(dt):
    global a
    global cell_body, cell_shape
    global cell_body2
    global angle1
    global angle2
    global cell_length, cell_length2, cell_width, cell_width2, position1, position2
    if a == 0:
        cell_length = 40
        cell_length2 = 40
        cell_width = 20
        cell_width2 = 20
        position1 = (200,320)
        position2 = (250,300)
        angle1 = 0.11
        angle2 = np.pi/4.2

    for body in space.bodies:
        space.remove(body)
    for poly in space.shapes:
        space.remove(poly)
    

    cell_body, cell_shape, cell_length, cell_width, cell_resolution, angle1 = create_cell(cell_length, cell_width, 20, angle1, position1)
    cell_length = cell_length + 1   
    #cell_width = cell_width + 0.24
    
    cell_body2, cell_shape2, cell_length2, cell_width2, cell_resolution2, angle2 = create_cell(cell_length, cell_width2, 20, angle2, position2)
    cell_length2 = cell_length2 + 1  
    #cell_width2 = cell_width2 + 0.1
    space.add(cell_body, cell_shape,cell_body2, cell_shape2)




    for x in range(500):
        space.step(dt)
    print("Angle = " + str(cell_body.angle))
    print("Position = " + str(cell_body.position))
    print("C of gravity = "+str(cell_body.center_of_gravity))
    angle1 = cell_body.angle
    angle2 = cell_body2.angle
    if a > 0:
        position1 = cell_body.position
        position2 = cell_body2.position
    if a%10 == 0:
        vertices11 = []
        for v in cell_shape.get_vertices():
            x,y = v.rotated(cell_shape.body.angle) + cell_shape.body.position
            vertices11.append((x,y))

        vertices22 = []
        for v in cell_shape2.get_vertices():
            x,y = v.rotated(cell_shape2.body.angle) + cell_shape2.body.position
            vertices22.append((x,y))
        vertices11 = np.array(vertices11)
        vertices22 = np.array(vertices22)
        plt.plot(vertices11[:,0], vertices11[:,1])
        plt.plot(vertices22[:,0], vertices22[:,1])

        centroid1 = cell_geometry.centroid(vertices11)
        plt.scatter(cell_body.position[0],cell_body.position[1], s=200)
                
                
        centroid2 = cell_geometry.centroid(vertices22)
        plt.scatter(cell_body2.position[0],cell_body2.position[1], s=200)
        plt.ylim(0,720)
        plt.xlim(0,720)
        plt.savefig("/home/georgeos/Documents/GitHub/SYMPTOMM2/figures/{}.png".format(a))
        plt.clf()

 
        #space.reindex_shapes_for_body(cell_body)
        #space.reindex_shapes_for_body(cell_body2)
    a = a + 1



#if __name__ == "__main__":
    #pyglet.clock.schedule_interval(update, 1/60)
    #pyglet.app.run()


# %%
for x in range(1000):
    update(1/60)

# %%

