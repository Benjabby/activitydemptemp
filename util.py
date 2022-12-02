import time
import numpy as np

def create_time_ref():
    return time.time_ns() - time.perf_counter_ns()

def get_timestamp(global_ref):
    return time.perf_counter_ns() + global_ref
    
def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)