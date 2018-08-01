import numpy as np
import math
import collections


def wrap_to_pi(a):
    if (a < -math.pi) or (a > math.pi):
        a = (a+math.pi) % (2*math.pi)-math.pi
    else:
        a = a
    return a


def wrap_to_pi_vec(a):
    res_1 = ((a + math.pi) % (2 * math.pi) - math.pi) * ((a < -math.pi) | (a > math.pi))
    res_2 = a * ~((a < -math.pi) | (a > math.pi))
    res = res_1+res_2
    return res


def wrap_to_2pi(a):
    if (a < 0) or (a > 2*math.pi):
        a = abs(a % (2*math.pi))
    else:
        a = a
    return a


def wrap_to_2pi_vec(a):
    res_1 = (abs(a % (2 * math.pi))) * ((a < 0) | (a > 2 * math.pi))
    res_2 = a * ~((a < 0) | (a > 2 * math.pi))
    res = res_1+res_2
    return res


def distance_cos_2d(p1, p2):
    dist = 1-math.cos(p1.Th-p2.Th)+np.linalg.norm([p1.Rho - p2.Rho])
    return dist


def distance_cos_2d_vec(p1, p2):
    sub = np.subtract(p1, p2)
    dist = 1-np.cos(sub[:, 0])+abs(sub[:, 1])
    return dist


def distance_wrap_2d(p1, p2):
    ad = abs(wrap_to_pi(p1[0]-p2[0]))
    ld = abs(p1[1] - p2[1])
    dist = math.sqrt(ad*ad+ld*ld)
    return dist


def distance_wrap_2d_vec(p1, p2):
    diff = np.subtract(p1, p2)
    r = np.hsplit(diff, 2)
    ar = r[0].flatten()
    lr = r[1].flatten()
    ad = abs(wrap_to_pi_vec(ar))
    ld = abs(lr)
    ad_ad = np.multiply(ad, ad)
    ld_ld = np.multiply(ld, ld)
    dist = np.sqrt(ad_ad + ld_ld)
    return dist


def distance_wrap_2d_vec_pair(p1, p2):

    shape1 = np.shape(p1)[0]
    shape2 = np.shape(p2)[0]
    p2 = p2.flatten()

    p1_rep = np.tile(p1, (1, shape2))
    p2_rep = np.tile(p2, (shape1, 1))

    diff = np.subtract(p1_rep, p2_rep)

    columns = int(np.shape(diff)[1]/2)
    diff_col = np.hsplit(diff, columns)
    dist = np.empty((0, columns), float)
    for c in range(0, columns):
        r = np.hsplit(diff_col[c], 2)
        ar = r[0].flatten()
        lr = r[1].flatten()
        ad = abs(wrap_to_pi_vec(ar))
        ld = abs(lr)
        ad_ad = np.multiply(ad, ad)
        ld_ld = np.multiply(ld, ld)
        d = np.array(np.sqrt(ad_ad + ld_ld))
        d = d.reshape(1, columns)
        dist = np.append(dist, d, axis=0)
    return dist


def distance_disjoint_2d(p1, p2):
    ad = abs(wrap_to_pi(p1.Th-p2.Th))
    ld = abs(p1.Rho - p2.Rho)
    distance = collections.namedtuple('distance', ['ad', 'ld'])
    dist = distance(ad, ld)
    return dist


def weighted_mean_2d_vec(p, w):
    a = p[:, 0]
    le = p[:, 1]

    c = np.sum(np.multiply(np.cos(a), w)) / np.sum(w)
    s = np.sum(np.multiply(np.sin(a), w)) / np.sum(w)

    if c >= 0:
        cr_m = np.arctan(s/c)
    else:
        cr_m = np.arctan(s/c)+math.pi
    l_m = np.sum(np.multiply(le, w)) / np.sum(w)
    mean = [wrap_to_2pi(cr_m), l_m]
    return mean
