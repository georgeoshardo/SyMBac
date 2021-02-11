import numpy as np
import pymunk
import sys
sys.path.insert(0,'/home/georgeos/Documents/GitHub/SYMPTOMM2/')
from SYMPTOMM import cell_geometry

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
        growth_rate_constant
    ):
        self.length = length
        self.width = width
        self.resolution = resolution
        self.angle = angle
        self.position = position
        self.space = space
        self.body, self.shape = self.create_pm_cell()
        self.angle = self.body.angle
        self.dt = dt
        self.growth_rate_constant = growth_rate_constant

    def create_pm_cell(self):
        cell_vertices = self.calculate_vertex_list()
        cell_shape = pymunk.Poly(None, cell_vertices)
        self.shape = cell_shape
        cell_moment = 1
        cell_mass = 0.00001
        cell_body = pymunk.Body(cell_mass,cell_moment)
        cell_shape.body = cell_body
        self.body = cell_body
        cell_body.position = self.position
        cell_body.angle = self.angle
        cell_shape.friction=1000
        self.space.add(cell_body, cell_shape)
        return cell_body, cell_shape

    def update_length(self):
        self.length = self.length + self.growth_rate_constant*self.dt*self.length

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
            x,y = v.rotated(self.shape.body.angle) + self.shape.body.position
            vertices.append((x,y))
        return vertices

    def get_centroid(self):
        vertices = self.get_vertex_list()
        return cell_geometry.centroid(vertices)

    