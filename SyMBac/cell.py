import numpy as np
import pymunk
import sys
#sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2/')
from SyMBac import cell_geometry
class Cell:
    """
    A class defining the cell objects
    
    ...
    
    Attributes
    ----------
    length : float
        Cell's length
    width : float
        Cell's width
    resolution : int
        Number of points defining cell's geometry
    position : (float, float)
        x,y coords of cell centroid
    angle : float
        rotation in radians of cell (counterclockwise)
    space : pymunk.space.Space
        The pymunk space of the cell
    dt : float
        Timestep the cell experiences every iteration
    growth_rate_constant : float
        The cell grows by a function of dt*growth_rate_constant depending on its growth model
    max_length : float
        The maximum length a cell reaches before dividing
    max_length_mean : float
        should be the same as max_length for reasons unless doing advanced simulations
    max_length_var : float
        The variance defining a normal distribution around max_length
    width_var : float
        The variance defining a normal distribution around width
    width_mean : float
        For reasons should be set equal to width unless using advanced features
    """
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
        
        """
        Initialising a cell

        Parameters
        ----------
        length : float
            Cell's length
        width : float
            Cell's width
        resolution : int
            Number of points defining cell's geometry
        position : (float, float)
            x,y coords of cell centroid
        angle : float
            rotation in radians of cell (counterclockwise)
        space : pymunk.space.Space
            The pymunk space of the cell
        dt : float
            Timestep the cell experiences every iteration
        growth_rate_constant : float
            The cell grows by a function of dt*growth_rate_constant depending on its growth model
        max_length : float
            The maximum length a cell reaches before dividing
        max_length_mean : float
            should be the same as max_length for reasons unless doing advanced simulations
        max_length_var : float
            The variance defining a normal distribution around max_length
        width_var : float
            The variance defining a normal distribution around width
        width_mean : float
            For reasons should be set equal to width unless using advanced features
        body : pymunk.body.Body
            The cell's pymunk body object
        shape : pymunk.shapes.Poly
            The cell's pymunk body object
        ID : int
            A unique identifier for each cell. At the moment just a number from 0 to 100_000_000 and cross fingers that we get no collisions. 
            
        For info about the Pymunk objects, see the API reference. http://www.pymunk.org/en/latest/pymunk.html
        Cell class has been tested and works with pymunk version 6.0.0
        """
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
            cell_shape.friction=1
            self.space.add(cell_body, cell_shape)
            daughter_details = {
                "length": daughter_length*0.9,
                "width": np.random.normal(self.width_mean,self.width_var),
                "resolution": self.resolution,
                "position": [self.position[0] - self.length/2 * np.cos(self.angle*2), self.position[1] - self.length/2 * np.sin(self.angle*2)],
                "angle": self.angle*np.random.uniform(0.95,1.05),
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
            cell_shape.friction=0.01
            self.space.add(cell_body, cell_shape)
            return cell_body, cell_shape

    def is_dividing(self): # This needs to be made constant or a cell can divide in one frame and not another frame
        if self.length > (self.max_length):
            return True
        else:
            return False


    def update_length(self):
        self.length = self.length + self.growth_rate_constant*self.dt*self.length*np.random.uniform(0.7,1.3)

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

    