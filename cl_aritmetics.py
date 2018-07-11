import numpy as np
import math
import collections


def wrap_to_pi(a):
    if (a < -math.pi) or (a > math.pi):
        a = abs((a+math.pi)%(2*math.pi)-math.pi)
    else:
        a = a
    return a

def wrap_to_pi_vec(a):
    res_1 = (abs((a + math.pi) % (2 * math.pi) - math.pi)) * ((a < 0) | (a > 2 * math.pi))
    res_2 = a * ~((a < 0) | (a > 2 * math.pi))
    res=res_1+res_2
    return res


def wrap_to_2pi(a):
    if (a < 0) or (a > 2*math.pi):
        a = abs((a+math.pi)%(2*math.pi))
    else:
        a = a
    return a


def wrap_to_2pi_vec(a):
    res_1 = (abs((a) % (2 * math.pi))) * ((a < 0) | (a > 2 * math.pi))
    res_2 = a * ~((a < 0) | (a > 2 * math.pi))
    res=res_1+res_2
    return res


def distance_cos_2d(p1, p2):
    dist = 1-math.cos(p1.Th-p2.Th)+np.linalg.norm([p1.Rho - p2.Rho])
    return dist


def distance_cos_2d_vec(p1, p2):
    sub=np.subtract(p1, p2)
    print(np.cos(sub[:, 0]))
    print(abs(sub[:, 1]))
    dist = 1-np.cos(sub[:, 0])+abs(sub[:, 1])
    return dist


def distance_wrap_2d(p1, p2):
    ad = abs(wrap_to_pi(p1[0]-p2[0]))
    ld = abs(p1[1] - p2[1])
    dist = math.sqrt(ad*ad+ld*ld)
    return dist


def distance_wrap_2d_vec(p1, p2):

    diff = np.subtract(p1, p2)
    print(diff)
    ad = abs(wrap_to_pi_vec(diff[:, 0]))
    ld = abs(diff[:, 1])
    dist = math.sqrt(ad*ad+ld*ld)
    return dist


def distance_disjoint_2d(p1, p2):
    ad = abs(wrap_to_pi(p1.Th-p2.Th))
    ld = abs(p1.Rho - p2.Rho)
    distance = collections.namedtuple('distance', ['ad', 'ld'])
    dist = distance(ad, ld)
    return dist

