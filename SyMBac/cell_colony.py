import numpy as np
import pymunk
import sys
#sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2/')
from SyMBac import cell_geometry

class Cell:
    def __init__(
        self,
        length,
        width,
        resolution,
        position,
        angle,
        space,
        dt,
        growth_rate_constant,
        max_length,
        max_length_mean,
        max_length_var,
        width_var,
        width_mean
    ):
        self.dt = dt
        self.growth_rate_constant = growth_rate_constant
        self.length = length
        self.width_mean = width_mean
        self.width_var = width_var
        self.width = width
        self.resolution = resolution
        self.angle = angle
        self.position = position
        self.space = space
        self.max_length = max_length
        self.max_length_mean = max_length_mean
        self.max_length_var = max_length_var
        print(self.create_pm_cell())
        self.body, self.shape = self.create_pm_cell()
        self.angle = self.body.angle
        self.ID = np.random.randint(0,100_000_000)
        
        
        
        

    def create_pm_cell(self):
        if self.is_dividing() == True:
            new_length = self.length/2
            daughter_length = self.length - new_length
            self.length = new_length
            cell_vertices = self.calculate_vertex_list()
            cell_shape = pymunk.Poly(None, cell_vertices)
            self.shape = cell_shape
            cell_moment = 1
            cell_mass = 0.0001
            cell_body = pymunk.Body(cell_mass,cell_moment)
            cell_shape.body = cell_body
            self.body = cell_body
            new_x = self.position[0] + self.length/2 * np.cos(self.angle*2)
            new_y = self.position[1] + self.length/2 * np.sin(self.angle*2)
            self.body.position = [new_x, new_y]
            cell_body.angle = self.angle
            cell_shape.friction=1000
            self.space.add(cell_body, cell_shape)
            daughter_details = {
                "length": daughter_length,
                "width": np.random.normal(self.width_mean,self.width_var),
                "resolution": self.resolution,
                "position": [self.position[0] - self.length/2 * np.cos(self.angle*2), self.position[1] - self.length/2 * np.sin(self.angle*2)],
                "angle": self.angle*np.random.uniform(0.85,1.15), # CHANGE THIS BACK FOR MM
                "space": self.space,
                "dt": self.dt,
                "growth_rate_constant": self.growth_rate_constant,
                "max_length": np.random.normal(self.max_length_mean,self.max_length_var),
                "max_length_mean": self.max_length_mean,
                "max_length_var": self.max_length_var,
                "width_var": self.width_var,
                "width_mean": self.width_mean
            }
            return daughter_details
        else:
            cell_vertices = self.calculate_vertex_list()
            cell_shape = pymunk.Poly(None, cell_vertices)
            self.shape = cell_shape
            cell_moment = 1
            cell_mass = 0.0001
            cell_body = pymunk.Body(cell_mass,cell_moment)
            cell_shape.body = cell_body
            self.body = cell_body
            cell_body.position = self.position
            cell_body.angle = self.angle
            cell_shape.friction=1000
            self.space.add(cell_body, cell_shape)
            return cell_body, cell_shape

    def is_dividing(self): # This needs to be made constant or a cell can divide in one frame and not another frame
        if self.length > (self.max_length):
            return True
        else:
            return False

    def growth_modifier(self, xpos, ypos):
        return 2 * (1/(1 + np.exp(-0.005 * abs(xpos-400))) - 0.5) * (1/(1 + np.exp(-0.1 * abs(ypos-400))) - 0.5) + 1/(len(self.space.bodies)+2)
    #This was changed for the colony
    
    def update_length(self):
        self.length = self.length + self.growth_rate_constant*self.dt*self.length*np.random.uniform(0.5,1.2)* self.growth_modifier(self.position[0], self.position[1])
        for _ in range(50+len(self.space.bodies)):
            self.space.step(self.dt)
    
    def update_position(self):
        self.position = self.body.position
        self.angle = self.body.angle
    
    def get_angle(self):
        return self.body.angle
    
    def calculate_vertex_list(self):
        return cell_geometry.get_vertices(
            self.length,
            self.width,
            self.angle, 
            self.resolution
            )

    def get_vertex_list(self):
        vertices = []
        for v in self.shape.get_vertices():
            x,y = v.rotated(self.shape.body.angle) + self.shape.body.position #.rotated(self.shape.body.angle)
            vertices.append((x,y))
        return vertices

    def get_centroid(self):
        vertices = self.get_vertex_list()
        return cell_geometry.centroid(vertices)

    