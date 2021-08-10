"""Neighbouring Contribution Function (new version updated Aug 10th 2021)

   * Assume a given x-grid xv[k], and a value x. For a 1D case, the Neighbouring Contribution Function gives ncf(k,x) = (x-xv[s])/dv for k=s, (xv[s]-x)/dv for k=s+1, and 0 elsewhere, where s is the nearest index of xv[k] to x but xv[k]<x. 
   * For a 2D case, NCF gives the non-zero values for 4 points around (x,y)

"""
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import vmap
from jax.lax import scan
from jax.ops import index_add
from jax.ops import index as joi

def getix(x,xv):
    indarr=jnp.arange(len(xv))
    pos = jnp.interp(x,xv,indarr)
    index = (pos).astype(int)
    cont = (pos-index)
    return cont,index

@jit
def inc1D(w,x,xv):
    cx,ix=getix(x,xv)
    a=jnp.zeros(len(xv))
    a=index_add(a,joi[ix],w*(1.0-cx))
    a=index_add(a,joi[ix+1],w*cx)
    return a


@jit
def inc2D(w,x,y,xv,yv):
    """integrated neighbouring contribution function for 2D (memory reduced sum).
    
    Args:
        w: weight (N)
        x: x values (N)
        y: y values (N)
        xv: x grid
        yv: y grid
            
    Returns:
        integrated neighbouring contribution function
        
    Note:
        This function computes \sum_n w_n fx_n \otimes fy_n, 
        where w_n is the weight, fx_n and fy_n are the n-th NCFs for 1D. 
        A direct sum uses huge RAM. 
        In this function, we use jax.lax.scan to compute the sum
        
    Example:
        >>> N=10000
        >>> xv=jnp.linspace(0,1,11) #grid
        >>> yv=jnp.linspace(0,1,13) #grid
        >>> w=np.logspace(1.0,3.0,N)
        >>> x=np.random.rand(N)
        >>> y=np.random.rand(N)
        >>> val=inc2D(w,x,y,xv,yv)
        >>> #the comparision with the direct sum
        >>> valdirect=jnp.sum(nc2D(x,y,xv,yv)*w,axis=2)        
        >>> #maximum deviation
        >>> print(jnp.max(jnp.abs((val-valdirect)/jnp.mean(valdirect)))*100,"%") #%
        >>> 5.196106e-05 %
        >>> #mean deviation
        >>> print(jnp.sqrt(jnp.mean((val-valdirect)**2))/jnp.mean(valdirect)*100,"%") #%
        >>> 1.6135311e-05 %
    """

    cx,ix=getix(x,xv)
    cy,iy=getix(y,yv)
    a=jnp.zeros((len(xv),len(yv)))
    a=index_add(a,joi[ix,iy],w*(1-cx)*(1-cy))
    a=index_add(a,joi[ix,iy+1],w*(1-cx)*cy)
    a=index_add(a,joi[ix+1,iy],w*cx*(1-cy))
    a=index_add(a,joi[ix+1,iy+1],w*cx*cy)
    return a

@jit
def inc3D(w,x,y,z,xv,yv,zv):
    """The lineshape distribution matrix = integrated neighbouring contribution for 3D (memory reduced sum).
    
    Args:
        w: weight (N)
        x: x values (N)
        y: y values (N)
        z: z values (N)
        xv: x grid
        yv: y grid
        zv: z grid            
        
    Returns:
        lineshape distribution matrix (integrated neighbouring contribution for 3D)
        
    Note:
        This function computes \sum_n w_n fx_n \otimes fy_n \otimes fz_n, 
        where w_n is the weight, fx_n, fy_n, and fz_n are the n-th NCFs for 1D. 
        A direct sum uses huge RAM. 
        In this function, we use jax.lax.scan to compute the sum
        
    Example:
        >>> N=10000
        >>> xv=jnp.linspace(0,1,11) #grid
        >>> yv=jnp.linspace(0,1,13) #grid
        >>> zv=jnp.linspace(0,1,17) #grid
        >>> w=np.logspace(1.0,3.0,N)
        >>> x=np.random.rand(N)
        >>> y=np.random.rand(N)
        >>> z=np.random.rand(N)
        >>> val=inc3D(w,x,y,z,xv,yv,zv)
        >>> #the comparision with the direct sum
        >>> valdirect=jnp.sum(nc3D(x,y,z,xv,yv,zv)*w,axis=3)
        >>> #maximum deviation
        >>> print(jnp.max(jnp.abs((val-valdirect)/jnp.mean(valdirect)))*100,"%") #%
        >>> 5.520862e-05 %
        >>> #mean deviation
        >>> print(jnp.sqrt(jnp.mean((val-valdirect)**2))/jnp.mean(valdirect)*100,"%") #%
        >>> 8.418057e-06 %
    """

    cx,ix=getix(x,xv)
    cy,iy=getix(y,yv)
    cz,iz=getix(z,zv)

    a=jnp.zeros((len(xv),len(yv),len(zv)))
    a=index_add(a,joi[ix,iy,iz],w*(1-cx)*(1-cy)*(1-cz))
    a=index_add(a,joi[ix,iy+1,iz],w*(1-cx)*cy*(1-cz))
    a=index_add(a,joi[ix+1,iy,iz],w*cx*(1-cy)*(1-cz))
    a=index_add(a,joi[ix+1,iy+1,iz],w*cx*cy*(1-cz))
    a=index_add(a,joi[ix,iy,iz+1],w*(1-cx)*(1-cy)*cz)
    a=index_add(a,joi[ix,iy+1,iz+1],w*(1-cx)*cy*cz)
    a=index_add(a,joi[ix+1,iy,iz+1],w*cx*(1-cy)*cz)
    a=index_add(a,joi[ix+1,iy+1,iz+1],w*cx*cy*cz)

    return a
