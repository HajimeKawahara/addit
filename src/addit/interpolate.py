import jax.numpy as jnp
from jax.numpy import searchsorted, clip, where
from jax import jit

@jit
def interp2d(x, y, xp, yp, fp):
    """interpolation of a 2D function f(x,y) from given precomputed x,y,f arrays.

    Args:
        x: x value to be evaluated
        y: y value to be evaluated
        xp: Arrays defining the data point coordinates for x
        yp: Arrays defining the data point coordinates for y
        fp: 2D array of the function to interpolate at the data points.

    Returns: 
        f(x,y)

    """

    i = clip(searchsorted(xp, x, side='right'), 1, len(xp) - 1)
    j = clip(searchsorted(yp, y, side='right'), 1, len(yp) - 1)
    dfx = fp[i,j] - fp[i - 1,j]
    dx = xp[i] - xp[i - 1]
    deltax = x - xp[i - 1]
    
    dfy = fp[i,j] - fp[i,j - 1]
    dy = yp[j] - yp[j - 1]
    deltay = y - yp[j - 1]
    
    f = fp[i - 1, j - 1] + (deltax / dx) * dfx + (deltay / dy) * dfy
    return f
