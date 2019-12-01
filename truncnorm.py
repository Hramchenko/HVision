from random import random, uniform, gauss
import scipy as sp

def truncnorm(mean, std, trunc_factor=2):
    boundary = std*trunc_factor
    while True:
        x = gauss(mean, std)
        if sp.fabs(x - mean) <= boundary:
            return x 
