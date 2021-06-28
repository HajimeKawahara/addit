"""Neighbouring Contribution Function 

   * Assume a given x-grid xv[k], and a value x. For a 1D case, the Neighbouring Contribution Function gives ncf(k,x) = (x-xv[s])/dv for k=s, (xv[s]-x)/dv for k=s+1, and 0 elsewhere, where s is the nearest index of xv[k] to x but xv[k]<x. 
   * For a 2D case, NCF gives the non-zero values for 4 points around (x,y)

"""

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import vmap
from jax.lax import scan

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
    """neighbouring contribution function for 2D.
    
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

def nc3D(x,y,z,xv,yv,zv):
    """neighbouring contribution function for 3D.
    
    Note:
        See Fig. 1 in van den Bekerom and Pannier (2021) JQSR 261, 107476

    Args:
        x: x value
        y: y value
        z: z value
        xv: x grid
        yv: y grid
        zv: z grid
            
    Returns:
        3D neighbouring contribution function
        
    """
    indarrx=jnp.arange(len(xv))
    indarry=jnp.arange(len(yv))
    indarrz=jnp.arange(len(zv))

    vcl=vmap(conti,(0,None,None),0)
    fx=vcl(indarrx,x,xv)
    fy=vcl(indarry,y,yv)
    fz=vcl(indarrz,z,zv)

    return fx[:,None,None]*fy[None,:,None]*fz[None,None,:]

@jit
def inc2D(x,y,xv,yv):
    """integrated neighbouring contribution function for 2D (memory reduced sum).
    
    Args:
        x: x values
        y: y values
        xv: x grid
        yv: y grid
            
    Returns:
        integrated neighbouring contribution function
        
    Note:
        This function computes \sum_n fx_n \otimes fy_n, 
        where fx_n and fy_n are the n-th NCFs for 1D. 
        A direct sum uses huge RAM. 
        In this function, we use jax.lax.scan to compute the sum
        
    Example:
        >>>N=10000
        >>> xv=jnp.linspace(0,1,11) #grid
        >>> yv=jnp.linspace(0,1,11) #grid
        >>> x=np.random.rand(N)
        >>> y=np.random.rand(N)
        >>> val=inc2D(x,y,xv,yv)
        The direct sum is computed as
        >>> valdirect=jnp.sum(nc2D(x,y,xv,yv),axis=2)
        >>> jnp.mean(jnp.sqrt((val/valdirect-1.0)**2))
        >>> DeviceArray(2.0836995e-07, dtype=float32)
        
    """
    Ngx=len(xv)
    Ngy=len(yv)
    indarrx=jnp.arange(Ngx)
    indarry=jnp.arange(Ngy)
    vcl=vmap(conti,(0,None,None),0)
    fx=vcl(indarrx,x,xv) # Ngx x N  memory
    fy=vcl(indarry,y,yv) # Ngy x N memory
    #jnp.sum(fx[:,None]*fy[None,:],axis=2) Ngx x Ngy x N -> huge memory 
    
    fxy=jnp.array([fx,fy]).T
    def fsum(x,arr):
        null=0.0
        fx=arr[:,0]
        fy=arr[:,1]
        val=x+fx[:,None]*fy[None,:]
        return val, null
    
    init0=jnp.zeros((Ngx,Ngy))
    val,null=scan(fsum,init0,fxy)
    return val

@jit
def inc3D(x,y,z,xv,yv,zv):
    """integrated neighbouring contribution for 3D (memory reduced sum).
    
    Args:
        x: x values
        y: y values
        z: z values
        xv: x grid
        yv: y grid
        zv: z grid            
        
    Returns:
        integrated neighbouring contribution 
        
    Note:
        This function computes \sum_n fx_n \otimes fy_n \otimes fz_n, 
        where fx_n, fy_n, and fz_n are the n-th NCFs for 1D. 
        A direct sum uses huge RAM. 
        In this function, we use jax.lax.scan to compute the sum
        
    Example:
        >>>N=10000
        >>> xv=jnp.linspace(0,1,11) #grid
        >>> yv=jnp.linspace(0,1,11) #grid
        >>> zv=jnp.linspace(0,1,11) #grid
        >>> x=np.random.rand(N)
        >>> y=np.random.rand(N)
        >>> z=np.random.rand(N)
        >>> val=inc3D(x,y,z,xv,yv,zv)
        The direct sum is computed as
        >>> valdirect=jnp.sum(nc3D(x,y,z,xv,yv,zv),axis=3) # direct sum (lots memory)
        >>> jnp.mean(jnp.sqrt((val/valdirect-1.0)**2))
        >>> DeviceArray(9.686315e-08, dtype=float32)
        
    """
    Ngx=len(xv)
    Ngy=len(yv)
    Ngz=len(zv)
    indarrx=jnp.arange(Ngx)
    indarry=jnp.arange(Ngy)
    indarrz=jnp.arange(Ngz)
    
    vcl=vmap(conti,(0,None,None),0)
    fx=vcl(indarrx,x,xv) # Ngx x N  memory
    fy=vcl(indarry,y,yv) # Ngy x N memory
    fz=vcl(indarrz,z,zv) # Ngz x N memory
    fxyz=jnp.array([fx,fy,fz]).T
    
    def fsum(x,arr):
        null=0.0
        fx=arr[:,0]
        fy=arr[:,1]
        fz=arr[:,2]
        val=x+fx[:,None,None]*fy[None,:,None]*fz[None,None,:]
        return val, null
    
    init0=jnp.zeros((Ngx,Ngy,Ngz))
    val,null=scan(fsum,init0,fxyz)
    return val
    
