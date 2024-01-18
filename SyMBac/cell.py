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
        self.angle = angle
        self.position = position
        self.space = space
        self.max_length = max_length
        self.max_length_mean = max_length_mean
        self.max_length_var = max_length_var
        self.body, self.shape = self.create_pm_cell()
        self.angle = self.body.angle
        self.ID = np.random.randint(0,100_000_000)
        self.lysis_p = lysis_p
        self.parent = parent
        self.daughter = daughter
        self.pinching_sep = pinching_sep
        

    def create_pm_cell(self):
        """
        Creates a pymunk (pm) cell object, and places it into the pymunk space given when initialising the cell. If the
        cell is dividing, then two cells will be created. Typically this function is called for every cell, in every
        timestep to update the entire simulation.

        .. note::
           The return type of this function is dependent on the value returned by :meth:`SyMBac.cell.Cell.is_dividing()`.
           This is not good, and will be changed in a future version.

        Returns
        -------
        dict or (pymunk.body, pymunk.shape)

           If :meth:`SyMBac.cell.Cell.is_dividing()` returns `True`, then a dictionary of values for the daughter cell
           is returned. A daughter can then be created. E.g:

           >>> daughter_details = cell.create_pm_cell()
           >>> daughter = Cell(**daughter_details)

           If :meth:`SyMBac.cell.Cell.is_dividing()` returns `False`, then only a tuple containing (pymunk.body, pymunk.shape) will be returned.
        """

        if self.is_dividing():
            self.length = self.length/2 * 0.98 # It just has to be done in this order due to when we call self.body.position = [new_x, new_y]
            daughter_length = self.length
            self.pinching_sep = 0
            
            cell_vertices = self.calculate_vertex_list()
            cell_shape = pymunk.Poly(None, cell_vertices)
            self.shape = cell_shape
            cell_mass = 0.000001
            cell_moment = pymunk.moment_for_poly(cell_mass, cell_shape.get_vertices())
            cell_body = pymunk.Body(cell_mass,cell_moment)
            cell_shape.body  = cell_body
            self.body = cell_body
            new_x = self.position[0] + self.length/2 *  np.cos(self.angle)
            new_y = self.position[1] + self.length/2 *  np.sin(self.angle)
            self.body.position = [new_x, new_y]
            cell_body.angle = self.angle
            cell_shape.friction=0
            #self.space.add(cell_body, cell_shape)
            daughter_details = {
                "length": daughter_length,
                "width": np.random.normal(self.width_mean,self.width_var),
                "resolution": self.resolution,
                "position": [self.position[0] - self.length/2 *  np.cos(self.angle), self.position[1] - self.length/2 *  np.sin(self.angle)],
                "angle": self.angle*np.random.uniform(0.95,1.05),
                "space": self.space,
                "dt": self.dt,
                "growth_rate_constant": self.growth_rate_constant,
                "max_length": np.random.normal(self.max_length_mean,self.max_length_var),
                "max_length_mean": self.max_length_mean,
                "max_length_var": self.max_length_var,
                "width_var": self.width_var,
                "width_mean": self.width_mean,
                "lysis_p": self.lysis_p,
                "parent": self.parent,
                "pinching_sep": 0
            }

            
            #self.position = [new_x, new_y]
            return daughter_details
        else:
            cell_vertices = self.calculate_vertex_list()
            cell_shape = pymunk.Poly(None, cell_vertices)
            self.shape = cell_shape
            cell_mass = 0.000001
            cell_moment = pymunk.moment_for_poly(cell_mass, cell_shape.get_vertices())
            cell_body = pymunk.Body(cell_mass,cell_moment)
            cell_shape.body = cell_body
            self.body = cell_body
            cell_body.position = self.position
            cell_body.angle = self.angle
            cell_shape.friction=0
            #self.space.add(cell_body, cell_shape)
            return cell_body, cell_shape

    def is_dividing(self): # This needs to be made constant or a cell can divide in one frame and not another frame
        """
        Checks whether a cell is dividing by comparing its current length to its max length (defined when instnatiated).

        Returns
        -------
        output : bool
            `True` if ``self.length > self.max_length``, else `False`.
        """
        if self.length > (self.max_length):
            return True
        else:
            return False


    def update_length(self):
        """
        A method, typically called every timepoint to update the length of the cell according to ``self.length = self.length + self.growth_rate_constant*self.dt*self.length``.

        Contains additional logic to control the amount of cell pinching happening according to the difference between
        the maximum length and the current length.

        Returns
        -------
        None
        """

        self.length = self.length + self.growth_rate_constant*self.dt*self.length*np.random.uniform(0.5,1.3)
        self.pinching_sep = max(0, self.length - self.max_length + self.width)
        self.pinching_sep = min(self.pinching_sep, self.width - 2)

    def update_position(self):
        """
        A method, typically called every timepoint to keep synchronised the cell position (``self.position`` and ``self.angle``)
        with the position of the cell's corresponding body in the pymunk space (``self.body.position`` and ``self.body.angle``).

        Returns
        -------
        None
        """
        self.position = self.body.position
        self.angle = self.body.angle

    def update_parent(self, parent):
        """
        Parameters
        ----------
        parent : :class:`SyMBac.cell.Cell`
           The SyMBac cell object to assign as the parent to the current cell.

        Returns
        -------
        None
        """
        self.parent = parent

    def get_angle(self):
        """
        Gets the angle of the cell's pymunk body.

        Returns
        -------
        angle : float
           The cell's angle in radians.
        """
        return self.body.angle
    
    def calculate_vertex_list(self):
        return cell_geometry.get_vertices(
            self.length,
            self.width,
            0,#self.angle, 
            self.resolution
            )

    def get_vertex_list(self):
        """
        Calculates the vertex list (a set of x,y coordinates) which parameterise the outline of the cell

        Returns
        -------
        vertices : list(tuple(float, float))
           A list of vertices, each in a tuple, where the order is `(x, y)`. The coordinates are relative to the pymunk
           space in which the cell exists.
        """
        vertices = []
        for v in self.shape.get_vertices():
            x,y = v.rotated(self.shape.body.angle) + self.shape.body.position #.rotated(self.shape.body.angle)
            vertices.append((x,y))
        return vertices

    def get_centroid(self):
        """
        Calculates the centroid of the cell from the vertices.

        Returns
        -------
        centroid : float
            The cell's centroid in coordinates relative to the pymunk space which the cell exists in.
        """
        vertices = self.get_vertex_list()
        return cell_geometry.centroid(vertices)

    
