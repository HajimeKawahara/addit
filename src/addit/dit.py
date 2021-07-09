import jax.numpy as jnp
from jax import jit
from addit.ncf import inc3D

def voigt_kernel(k, beta,gammaL):
    """Fourier Kernel of the Voigt Profile
    
    Args:
        k: conjugated of wavenumber
        beta: Gaussian standard deviation
        gammaL: Lorentian Half Width
        
    Returns:
        kernel (N_x,N_beta,N_gammaL)
    
    Note:
        Conversions to the (full) width, wG and wL are as follows: 
        wG=2*sqrt(2*ln2) beta
        wL=2*gamma
    
    """
    val=(jnp.pi*beta[None,:,None]*k[:,None,None])**2 + jnp.pi*gammaL[None,None,:]*k[:,None,None]
    return jnp.exp(-2.0*val)

@jit
def rundit(S,nu_lines,beta,gammaL,nu_grid,beta_grid,gammaL_grid):
    """run DIT
    
    Args:
       S: line strength (Nlines)
       nu_lines: line center (Nlines)
       beta: Gaussian STD (Nlines)
       gammaL: Lorentian half width (Nlines)
       nu_grid: linear wavenumber grid
       beta_grid: beta grid
       gammaL_grid: gammaL grid

    Returns:
       Cross section

    """
    Ng_nu=len(nu_grid)
    Ng_beta=len(beta_grid)
    Ng_gammaL=len(gammaL_grid)
    
    log_beta=jnp.log(beta)
    log_gammaL=jnp.log(gammaL)
    
    log_beta_grid = jnp.log(beta_grid)
    log_gammaL_grid = jnp.log(gammaL_grid)
    
    dnu = (nu_grid[-1]-nu_grid[0])/(Ng_nu-1)
    k = jnp.fft.rfftfreq(2*Ng_nu,dnu)
    val=inc3D(S,nu_lines,log_beta,log_gammaL,nu_grid,log_beta_grid,log_gammaL_grid)
    valbuf=jnp.vstack([val,jnp.zeros_like(val)])
    fftval = jnp.fft.rfft(valbuf,axis=0)
    vk=voigt_kernel(k, beta_grid,gammaL_grid)
    fftvalsum = jnp.sum(fftval*vk,axis=(1,2))
    #F0=jnp.fft.irfft(fftvalsum)[:Ng_nu]
    F0=jnp.fft.irfft(fftvalsum)[:Ng_nu]/dnu
    return F0, val

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from exojax.spec import xsection
    import numpy as np
#    N=10000
    N=1
    Ng_nu=10000
    Ng_beta=11
    Ng_gammaL=11
    dv=100
    nus=np.linspace(2050.0-dv,2250.0+dv,Ng_nu) #nu grid
    beta_grid=np.logspace(-1,0,Ng_beta) #beta grid
    gammaL_grid=np.logspace(1,2,Ng_gammaL)#gammaL grid
    S=np.logspace(0.0,2.0,N)
    S=np.zeros(N)
    S[0:10]=1000.0

    np.random.seed(20)
    nu_lines=np.random.rand(N)*(nus[-1]-nus[0]-50.0)+nus[0]+25.0

    ####
#    beta=np.random.rand(N)*0.3333
#    gammaL=np.random.rand(N)*3.33333
    beta=np.ones(N)*0.3333
    gammaL=np.ones(N)*33.333
    ####
    #beta=np.ones(N)*0.1
    #gammaL=np.ones(N)*100.0
    ####
    
    F0,val=rundit(S,nu_lines,beta,gammaL,nus,beta_grid,gammaL_grid)

    print("-beta-------------")
    print(beta)
    print(beta_grid)
    print("-gamma-------------")
    print(gammaL)
    print(gammaL_grid)
#    print(np.sum(val,axis=0))
    plt.imshow(np.sum(val,axis=0))
    plt.show()
#    import sys
#    sys.exit()

    xsv=xsection(nus,nu_lines,beta,gammaL,S)

    fig=plt.figure()
    fig.add_subplot(211)
    plt.plot(nus,xsv,label="exojax")
    plt.plot(nus,F0,label="addit")
    plt.legend()
    for i in range(0,1):
        plt.axvline(nu_lines[i])
        plt.text(nu_lines[i],10.0,str(i))
    fig.add_subplot(212)
    plt.plot(nus,(xsv-F0),label="exojax-addit")
    plt.show()
