import jax.numpy as jnp
from jax import jit
from addit.ncf import inc2D
from jax.lax import scan


def folded_voigt_kernel_logst(k,log_nstbeta,log_ngammaL,dLarray):
    """Folded Fourier Kernel of the Voigt Profile
    
    Args:
        k: conjugate wavenumber
        log_nstbeta: log normalized Gaussian standard deviation (scalar)
        log_ngammaL: log normalized Lorentian Half Width (Nlines)
        dLarray: dLarray
        
    Returns:
        kernel (N_x,N_gammaL)
    
    Note:
        Conversions to the (full) width, wG and wL are as follows: 
        wG=2*sqrt(2*ln2) beta
        wL=2*gamma
    
    """

    beta=jnp.exp(log_nstbeta)
    gammaL=jnp.exp(log_ngammaL)
#    print(jnp.max(k))
#    print(dLarray)
    def ffold(val,dL):
        val=val+jnp.exp(-2.0*((jnp.pi*beta*(k[:,None]+dL))**2 \
                              + jnp.pi*gammaL[None,:]*(k[:,None]+dL)))
        val=val+jnp.exp(-2.0*((jnp.pi*beta*(k[:,None]-dL))**2 \
                              + jnp.pi*gammaL[None,:]*(dL-k[:,None])))
        null=0.0
        return val, null
    print(jnp.shape(k),jnp.shape(gammaL),jnp.shape(beta))
    val=jnp.exp(-2.0*((jnp.pi*beta*k[:,None])**2 + jnp.pi*gammaL[None,:]*k[:,None]))
    
    val,nullstack=scan(ffold, val, dLarray)
    
    return val


#@jit
def rundit_fold_logredst(S,nu_lines,cnbeta,gammaL,nu_grid,ngammaL_grid,dLarray,dv_lines,dv_grid):
    """run DIT folded voigt for ESLOG for reduced wavenumebr inputs (against the truncation error) for a constant beta

    Args:
       S: line strength (Nlines)
       nu_lines: (reduced) line center (Nlines)
       cnbeta: constant normalized Gaussian STD 
       gammaL: Lorentian half width (Nlines)
       nu_grid: (reduced) evenly spaced log (ESLOG) wavenumber grid
       ngammaL_grid: normalized gammaL grid
       dLarray: dLarray
       dv_lines: delta wavenumber for lines i.e. nu_lines/R
       dv_grid: delta wavenumber for nu_grid i.e. nu_grid/R

    Returns:
       Cross section

    
    """



    Ng_nu=len(nu_grid)
    Ng_gammaL=len(ngammaL_grid)

    ngammaL=gammaL/dv_lines
    log_nstbeta=jnp.log(cnbeta)
    log_ngammaL=jnp.log(ngammaL)
    
    log_ngammaL_grid = jnp.log(ngammaL_grid)

    k = jnp.fft.rfftfreq(2*Ng_nu,1)
    val=inc2D(S,nu_lines,log_ngammaL,nu_grid,log_ngammaL_grid)
    valbuf=jnp.vstack([val,jnp.zeros_like(val)])
    fftval = jnp.fft.rfft(valbuf,axis=0)
    vk=folded_voigt_kernel_logst(k, log_nstbeta,log_ngammaL_grid,dLarray)
    fftvalsum = jnp.sum(fftval*vk,axis=(1,))
    xs=jnp.fft.irfft(fftvalsum)[:Ng_nu]/dv_grid
    
    return xs
