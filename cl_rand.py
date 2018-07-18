import numpy as np
import cl_arithmetic as cl_a


def cl_gauss_2d(mu, sigma, n):
    ret = np.random.multivariate_normal(mu, sigma, n)
    ret[:,0]=cl_a.wrap_to_2pi_vec(ret[:, 0])
    return ret
