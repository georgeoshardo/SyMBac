import numpy as np
import pymunk
from SyMBac import cell_geometry


class Cell:
    def __init__(self,
    length, 
    width, 
    x_pos, 
    y_pos,
    angle = 0, 
    resolution = 30, 
    model = "adder", 
    food_conc = 5, 
    mass = 0.000001, 
    friction = 0, 
    pm_object = 0, 
    division_threshold = 30,
    mother = None):
        self.length = length
        self.width = width
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.angle = angle
        self.resolution = resolution
        self.model = model
        self.food_conc = food_conc
        self.mass = mass
        self.friction = friction
        self.pm_object = self.initialise_pm_object()
        self.division_threshold = division_threshold
        self.mother = mother
        
        
    
    def get_cell_vertices_for_draw(self, cell_length, cell_width, resolution):
    
        def circ(theta, start, radius):
            y = radius * np.cos(theta) +radius
            x = radius * np.sin(theta) + start + radius
            return x, y


        def wall(radius, start, end, t_or_b):
            wall_x = np.linspace(start, end, num = resolution)
            wall_y = np.ones(resolution)*radius * t_or_b +radius
            return wall_x, wall_y

        cell_width = cell_width/2
        cell_length = cell_length - cell_width
        left_wall = circ(np.linspace(np.pi,2*np.pi, num=resolution), 0, cell_width)
        top_wall_xy = wall(cell_width, cell_width, cell_length, 1)
        bottom_wall_xy = wall(cell_width, cell_width, cell_length, -1)
        right_wall = circ(np.linspace(0,np.pi, num=resolution), cell_length - cell_width, cell_width)
        return [[left_wall[0][x] - cell_length/2, left_wall[1][x] - cell_width/2] for x in reversed(range(len(left_wall[0])))] + \
                [[bottom_wall_xy[0][x] - cell_length/2, bottom_wall_xy[1][x]- cell_width/2] for x in (range(len(bottom_wall_xy[0])))] + \
                [[right_wall[0][x] - cell_length/2, right_wall[1][x]- cell_width/2] for x in reversed(range(len(right_wall[0])))] + \
                [[top_wall_xy[0][x] - cell_length/2, top_wall_xy[1][x]- cell_width/2] for x in reversed(range(len(top_wall_xy[0])))]

    
    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy


    def vertices(self):
        points = np.array(self.get_cell_vertices_for_draw(self.length, self.width, self.resolution))
        points[:,0] = points[:,0] + self.x_pos
        points[:,1] = points[:,1] + self.y_pos

        rotated = np.zeros((len(points),2))
        for x in range(len(points)):
            rotated[x] = self.rotate((self.x_pos, self.y_pos), (points[x][0],points[x][1]), self.angle)

        return rotated

    def growth_rate(self):
        return self.food_conc


    def growth(self, dt):
        if self.model == "adder":
            self.length = self.length + dt * self.growth_rate()
            self.is_dividing()
            self.x_pos, self.y_pos = self.pm_object[1].position[0], self.pm_object[1].position[1]
            self.angle = self.angle +  self.pm_object[1].angle
    
    def initialise_pm_object(self):
        _poly_vertices = [tuple(vertex) for vertex in self.vertices().tolist()]
        _poly = pm.Poly(None, _poly_vertices)

        _poly.friction = self.friction
        _moment = pm.moment_for_poly(self.mass, _poly.get_vertices())
        _body = pm.Body(self.mass, _moment)
        _body._set_angle = self.angle
        _poly.body = _body
        
        _body.position = self.x_pos, self.y_pos
        
        return (_poly, _body)

    def update_pm_object(self):
        self.pm_object = self.initialise_pm_object()


    def is_dividing(self):
        if self.length > self.division_threshold * np.random.uniform(low = 0.9, high = 1.1):
            return 1
        if self.length < self.division_threshold:
            return 0
    def divide(self):
        self.length = self.length/2
        self.x_pos = self.x_pos - self.length/4 * np.cos(self.angle)
        self.y_pos = self.y_pos - self.length/4 * np.sin(self.angle)
        self.angle = np.random.uniform(low = self.angle - 2*np.pi*0.03, high = self.angle + 2*np.pi*0.03)
    def centroid(self):
        length = self.vertices().shape[0]
        sum_x = np.sum(self.vertices()[:, 0])
        sum_y = np.sum(self.vertices()[:, 1])
        return sum_x/length, sum_y/length

    def daughter_length(self):
        return self.length
    def daughter_width(self):
        return np.random.uniform(low = self.width*0.95, high = self.width*1.05)
    def daughter_x_pos(self):
        return self.x_pos + self.length/2 * np.cos(self.angle)
    def daughter_y_pos(self):
        return self.y_pos + self.length/2 * np.sin(self.angle)
    def daughter_angle(self):
        return np.random.uniform(low = self.angle - 2*np.pi*0.03, high = self.angle + 2*np.pi*0.03)