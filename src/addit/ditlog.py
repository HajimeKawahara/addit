import jax.numpy as jnp
from jax import jit
from addit.ncf import inc3D
from jax.lax import scan


def folded_voigt_kernel_log(k,log_nbeta,log_ngammaL,dLarray):
    """Folded Fourier Kernel of the Voigt Profile
    
    Args:
        k: conjugate wavenumber
        beta: Gaussian standard deviation
        gammaL: Lorentian Half Width
        dLarray: dLarray
        
    Returns:
        kernel (N_x,N_beta,N_gammaL)
    
    Note:
        Conversions to the (full) width, wG and wL are as follows: 
        wG=2*sqrt(2*ln2) beta
        wL=2*gamma
    
    """

    beta=jnp.exp(log_nbeta)
    gammaL=jnp.exp(log_ngammaL)
#    print(jnp.max(k))
#    print(dLarray)
    def ffold(val,dL):
        val=val+jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*(k[:,None,None]+dL))**2 \
                              + jnp.pi*gammaL[None,None,:]*(k[:,None,None]+dL)))
        val=val+jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*(k[:,None,None]-dL))**2 \
                              + jnp.pi*gammaL[None,None,:]*(dL-k[:,None,None])))
        null=0.0
        return val, null
    
    val=jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*k[:,None,None])**2 + jnp.pi*gammaL[None,None,:]*k[:,None,None]))
    
    val,nullstack=scan(ffold, val, dLarray)
    
    return val



@jit
def rundit_fold_log(S,nu_lines,beta,gammaL,nu_grid,R,nbeta_grid,ngammaL_grid,dLarray):
    """run DIT folded voigt for an arbitrary ESLOG

    Args:
       S: line strength (Nlines)
       nu_lines: line center (Nlines)
       beta: Gaussian STD (Nlines)
       gammaL: Lorentian half width (Nlines)
       nu_grid: evenly spaced log (ESLOG) wavenumber grid
       R: spectral resolution
       nbeta_grid: normalized beta grid 
       ngammaL_grid: normalized gammaL grid
       dLarray: dLarray

    Returns:
       Cross section

    
    """
    Ng_nu=len(nu_grid)
    Ng_beta=len(nbeta_grid)
    Ng_gammaL=len(ngammaL_grid)

    nbeta=beta/nu_lines*R
    ngammaL=gammaL/nu_lines*R
    
    log_nbeta=jnp.log(nbeta)
    log_ngammaL=jnp.log(ngammaL)
    
    log_nbeta_grid = jnp.log(nbeta_grid)
    log_ngammaL_grid = jnp.log(ngammaL_grid)

    k = jnp.fft.rfftfreq(2*Ng_nu,1)
    val=inc3D(S,nu_lines,log_nbeta,log_ngammaL,nu_grid,log_nbeta_grid,log_ngammaL_grid)
    valbuf=jnp.vstack([val,jnp.zeros_like(val)])
    fftval = jnp.fft.rfft(valbuf,axis=0)
    vk=folded_voigt_kernel_log(k, log_nbeta_grid,log_ngammaL_grid,dLarray)
    fftvalsum = jnp.sum(fftval*vk,axis=(1,2))

    dv=nu_grid/R
    xs=jnp.fft.irfft(fftvalsum)[:Ng_nu]/dv

    return xs

