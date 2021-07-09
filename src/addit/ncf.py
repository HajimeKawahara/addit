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
def Xncf(i,x,xv):
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
    vcl=vmap(Xncf,(0,None,None),0)
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
    vcl=vmap(Xncf,(0,None,None),0)
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

    vcl=vmap(Xncf,(0,None,None),0)
    fx=vcl(indarrx,x,xv)
    fy=vcl(indarry,y,yv)
    fz=vcl(indarrz,z,zv)

    return fx[:,None,None]*fy[None,:,None]*fz[None,None,:]

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
    Ngx=len(xv)
    Ngy=len(yv)
    indarrx=jnp.arange(Ngx)
    indarry=jnp.arange(Ngy)
    vcl=vmap(Xncf,(0,None,None),0)
    fx=vcl(indarrx,x,xv) # Ngx x N  memory
    fy=vcl(indarry,y,yv) # Ngy x N memory
    #jnp.sum(fx[:,None]*fy[None,:],axis=2) Ngx x Ngy x N -> huge memory 
    fxy_w=jnp.vstack([fx,fy,w]).T
    
    def fsum(x,arr):
        null=0.0
        fx=arr[0:Ngx]
        fy=arr[Ngx:Ngx+Ngy]
        w=arr[Ngx+Ngy]
        val=x+w*fx[:,None]*fy[None,:]
        return val, null
    
    init0=jnp.zeros((Ngx,Ngy))
    val,null=scan(fsum,init0,fxy_w)
    return val

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
    Ngx=len(xv)
    Ngy=len(yv)
    Ngz=len(zv)
    indarrx=jnp.arange(Ngx)
    indarry=jnp.arange(Ngy)
    indarrz=jnp.arange(Ngz)
    
    vcl=vmap(Xncf,(0,None,None),0)
    fx=vcl(indarrx,x,xv) # Ngx x N  memory
    fy=vcl(indarry,y,yv) # Ngy x N memory
    fz=vcl(indarrz,z,zv) # Ngz x N memory

    fxyz_w=jnp.vstack([fx,fy,fz,w]).T
    def fsum(x,arr):
        null=0.0
        fx=arr[0:Ngx]
        fy=arr[Ngx:Ngx+Ngy]
        fz=arr[Ngx+Ngy:Ngx+Ngy+Ngz]
        w=arr[Ngx+Ngy+Ngz]
        val=x+w*fx[:,None,None]*fy[None,:,None]*fz[None,None,:]
        return val, null
    
    init0=jnp.zeros((Ngx,Ngy,Ngz))
    val,null=scan(fsum,init0,fxyz_w)
    return val
    
