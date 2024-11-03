import numpy as np
from numpy.typing import NDArray 


def axis_to_angle(axis: int, coord: NDArray[np.float_]) -> NDArray:        
    """
    Converts a given axis upon a cartesian plane to its polar coordinate
    """
    n = len(coord)
   
    # For all subsequent angles from 0 to N-2 capture the orientation of the
    # point relative to the other dimensions hiearchically (AKA in order). This
    # is done by taking the normal in axis+1-dimensional space (direction of 
    # point) by the value on the current axis.
    # This approach aligns with the way polar coordinates are traditionally
    # generate hyperspherical coordinate systems. By measuring each angle 
    # progressively. 
    norm_vec = np.linalg.norm(coord[:axis+1])
    if axis != n - 1:
        return np.arccos(coord[axis] / norm_vec)
    
    # Normally you would take arccos for all, but one angle is traditionally
    # reserved for the azimuthal angle. This angle handles direction on the N-1
    # plane. 
    
    # Important to note that since we are probably using this function 
    # limited to 2D space, the azimuthal angle is theta. 
    # NOTE: So please don't optimize this like a game engine!
    return np.arctan2(coord[axis], coord[axis - 1])

def cartesian_to_polar_nd(coord: NDArray[np.float_]):
    """
    Converts N dimensional cartesian coordinate to N dimensional polar 
    coordinates
    """

    n = len(coord)
    r = np.linalg.norm(coord)  # Calculate radius    

    return r, [axis_to_angle(i, coord) for i in range(1, n)]

def polar_to_cartesian_2d(r: float,theta: float):
    """
    Convert 2 dimensional polar coordinate to 2 dimensional cartesian coordinate
    """
    return (r * np.cos(theta), r * np.sin(theta))

def rel_op(coord0: NDArray[np.float_], coordN: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Shift a cartesian coordinate `coord0` to be centered at 0, then shift 
    `coordN` the same way to keep them relative.
    """
    return coordN - coord0.min()

import numpy as np
    
def coords_to_rad_vspace(v1_start, v1_end, v2_start, v2_end):
    """
    Calculates the radians based on two sets of coordinates and determines if the angle is 
    clockwise (right) or counter-clockwise (left) relative to v1
    
    v1 = ((x1, y1), (x2, y2))
    v2 = ((x1, y1), (x2, y2))
    
    Returns the angle in radians and the direction ('left' or 'right')
    """
    # Define vectors
    v1 = np.array([v1_end[0] - v1_start[0], v1_end[1] - v1_start[1]])
    v2 = np.array([v2_end[0] - v2_start[0], v2_end[1] - v2_start[1]])

    # Calculate angle in radians
    magnitude = np.linalg.norm
    angle = np.arccos(
        np.dot(v1, v2) / (magnitude(v1) * magnitude(v2))
    )
    
    # Determine direction (left or right)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    direction = "left" if cross_product > 0 else "right"

    return angle, direction

def sigmoid(x):
    return 1 / (1 + np.exp(-x))