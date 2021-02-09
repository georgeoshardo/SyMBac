import math
import numpy as np
import matplotlib.pyplot as plt
def circ(theta, start, radius):
    y = radius * np.cos(theta) +radius
    x = radius * np.sin(theta) + start + radius
    return x, y

def wall(radius, start, end, t_or_b, resolution):
    wall_x = np.linspace(start, end, resolution)
    wall_y = np.ones(resolution)*radius * t_or_b +radius
    return wall_x, wall_y

def get_vertices(cell_length, cell_width, angle, resolution):

    cell_width = cell_width/2
    left_wall = circ(np.linspace(np.pi,2*np.pi, resolution), 0, cell_width)
    top_wall_xy = wall(cell_width, cell_width, cell_length, 1, resolution)
    bottom_wall_xy = wall(cell_width, cell_width, cell_length, -1, resolution)
    right_wall = circ(np.linspace(0,np.pi, resolution), cell_length - cell_width, cell_width)
    coordinates = [[left_wall[0][x] - cell_length/2, left_wall[1][x] - cell_width/2] for x in reversed(range(len(left_wall[0])))] + \
            [[bottom_wall_xy[0][x] - cell_length/2, bottom_wall_xy[1][x]- cell_width/2] for x in (range(len(bottom_wall_xy[0])))] + \
            [[right_wall[0][x] - cell_length/2, right_wall[1][x]- cell_width/2] for x in reversed(range(len(right_wall[0])))] + \
            [[top_wall_xy[0][x] - cell_length/2, top_wall_xy[1][x]- cell_width/2] for x in reversed(range(len(top_wall_xy[0])))]
    coordinates = np.array(coordinates)
    cell_centroid = centroid(coordinates)
    centered_verts = coordinates - cell_centroid
    centered_verts = centered_verts.tolist()

    rotated = np.zeros((len(centered_verts),2))
    for x in range(len(centered_verts)):
        rotated[x] = rotate(cell_centroid, (centered_verts[x][0],centered_verts[x][1]), angle)
    centered_verts = rotated - centroid(rotated)

    return centered_verts.tolist()

def centroid(vertices: list[tuple]) -> tuple:
    """Return the centroid of a list of vertices 
    
    Keyword arguments:
    vertices -- A list of tuples containing x,y coordinates.

    """
    return np.sum(vertices,axis=0)/len(vertices)

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

if __name__ == "__main__":
    vertices = get_vertices(20, 10, 1, 20)
    vertices = np.array(vertices)
    plt.plot(vertices[:,0], vertices[:,1])
    plt.show()
    print(centroid(vertices))