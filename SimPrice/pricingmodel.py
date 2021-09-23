
import numpy as np


def get_price(x, c=1000., A=2000., epsilon=0.001, exponent=0.5):
    return A*np.log(1.+x/c) + epsilon*np.power(x, exponent)*np.random.normal()
