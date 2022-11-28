import numpy as np
import pymunk
from SyMBac import cell_geometry
class Cell:
    """
    Cells are the agents in the simulation. This class allows for instantiating `Cell` object.

    .. note::
       Typically the user will not need to call this class, as it will be handled by :meth:`SyMBac.cell_simulation`,
       specifically all cell setup
       happens when instantiating a simulation using :meth:`SyMBac.simulation.Simulation`
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
        width_mean,
        parent = None,
        daughter = None,
        lysis_p = 0,
        pinching_sep = 0
    ):
        
        """
        Initialising a cell

        For info about the Pymunk objects, see the API reference. http://www.pymunk.org/en/latest/pymunk.html Cell class has been tested and works with pymunk version 6.0.0

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
            new_length = self.length/2  - self.width/4
            daughter_length = self.length - new_length  - self.width/4
            self.length = new_length
            self.pinching_sep = 0
            cell_vertices = self.calculate_vertex_list()
            cell_shape = pymunk.Poly(None, cell_vertices)
            self.shape = cell_shape
            cell_moment = 100001
            cell_mass = 1
            cell_body = pymunk.Body(cell_mass,cell_moment)
            cell_shape.body = cell_body
            self.body = cell_body
            new_x = self.position[0] + (self.length + self.width/2)/2 * np.cos(self.angle*2)
            new_y = self.position[1] + (self.length+ self.width/2)/2 * np.sin(self.angle*2)
            self.body.position = [new_x, new_y]
            cell_body.angle = self.angle
            cell_shape.friction=0
            self.space.add(cell_body, cell_shape)
            daughter_details = {
                "length": daughter_length,
                "width": np.random.normal(self.width_mean,self.width_var),
                "resolution": self.resolution,
                "position": [self.position[0] - (self.length+ self.width/2)/2 * np.cos(self.angle*2), self.position[1] - (self.length+ self.width/2)/2 * np.sin(self.angle*2)],
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
            return daughter_details
        else:
            cell_vertices = self.calculate_vertex_list()
            cell_shape = pymunk.Poly(None, cell_vertices)
            self.shape = cell_shape
            cell_moment = 100001
            cell_mass = 1
            cell_body = pymunk.Body(cell_mass,cell_moment)
            cell_shape.body = cell_body
            self.body = cell_body
            cell_body.position = self.position
            cell_body.angle = self.angle
            cell_shape.friction=0
            self.space.add(cell_body, cell_shape)
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
            self.angle, 
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

    
