from collections import namedtuple
import numpy as np
CLDataPoint = namedtuple("CLDataPoint", "X Y Z Theta Gamma Rho T")


def cos_distance_2d(p1, p2):
    dist = 1-math.cos(p1.Theta-p2.Theta)+np.linalg.norm([p1.Rho, p2.Rho])
    return dist

