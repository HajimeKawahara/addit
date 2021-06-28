"""Neighbouring Contribution Function 

   * Assume a given x-grid xv[k], and a value x. For a 1D case, the Neighbouring Contribution Function gives ncf(k,x) = (x-xv[s])/dv for k=s, (xv[s]-x)/dv for k=s+1, and 0 elsewhere, where s is the nearest index of xv[k] to x but xv[k]<x. 
   * For a 2D case, NCF gives the non-zero values for 4 points around (x,y)

"""

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import vmap

@jit
def conti(i,x,xv):
    """neighbouring contribution function for index i.  
    
    Args:
        i: index 
        x: x value
        xv: x-grid
            
    Returns:
        neighbouring contribution function of x to the i-th component of the array with the same dimension as xv.
            
    """
    indarr=jnp.arange(len(xv))
    pos = jnp.interp(x,xv,indarr)
    index = (pos).astype(int)
    cont = pos-index
    f=jnp.where(index==i,1.0-cont,0.0)
    g=jnp.where(index+1==i,cont,0.0)
    return f+g


def nc1D(x,xv):
    """neighbouring contribution function for 1D.
    
    Args:
        x: x value
        xv: x grid
            
    Returns:
        neighbouring contribution function
        
    Example:
       >>> xv=jnp.linspace(0,1,11) #grid
       >>> print(nc1D(0.23,xv))
       >>> [0.         0.         0.70000005 0.29999995 0.         0.  0.         0.         0.         0.         0.        ]

    """
    indarr=jnp.arange(len(xv))
    vcl=vmap(conti,(0,None,None),0)
    return vcl(indarr,x,xv)

def nc2D(x,y,xv,yv):
    """neighbouring contribution for 2D.
    
    Args:
        x: x value
        y: y value
        xv: x grid
        yv: y grid
            
    Returns:
        2D neighbouring contribution function
        
    """
    indarrx=jnp.arange(len(xv))
    indarry=jnp.arange(len(yv))
    vcl=vmap(conti,(0,None,None),0)
    fx=vcl(indarrx,x,xv)
    fy=vcl(indarry,y,yv)
    return fx[:,None]*fy[None,:]
