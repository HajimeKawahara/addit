import jax.numpy as jnp
from jax import jit
from addit.ncf import inc3D
from jax.lax import scan
from jax.ops import index_add

def make_dLarray(Nfold,dnu):
    """compute dLarray for the DIT folding
    
    Args:
       Nfold: # of the folding
       dnu: linear wavenumber grid interval

    Returns:
       dLarray: ifold/dnu (ifold=1,..,Nfold) array

    """
    dLarray=jnp.linspace(1,Nfold,Nfold)/dnu                
    return dLarray


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
    val=(jnp.pi*beta[None,:,None]*k[:,None,None])**2 + 4*jnp.pi*gammaL[None,None,:]*k[:,None,None]
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
    val=inc3D(S,nu_lines,log_beta,log_gammaL,nu_grid,log_beta_grid,log_gammaL_grid)
    #Nbuf=1
    #k = jnp.fft.rfftfreq(Ng_nu,dnu)
    #valbuf=val
    #Nbuf=2
    k = jnp.fft.rfftfreq(2*Ng_nu,dnu)
    valbuf=jnp.vstack([val,jnp.zeros_like(val)])
    #Nbuf=4
    #k = jnp.fft.rfftfreq(4*Ng_nu,dnu)
    #valbuf=jnp.vstack([val,jnp.zeros_like(val),jnp.zeros_like(val),jnp.zeros_like(val)])
    fftval = jnp.fft.rfft(valbuf,axis=0)
    vk=voigt_kernel(k, beta_grid,gammaL_grid)
    fftvalsum = jnp.sum(fftval*vk,axis=(1,2))
    xs=jnp.fft.irfft(fftvalsum)[:Ng_nu]/dnu
    return xs

@jit
def runditnewfold(S,nu_lines,beta,gammaL,nu_grid,beta_grid,gammaL_grid):
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
    val=inc3D(S,nu_lines,log_beta,log_gammaL,nu_grid,log_beta_grid,log_gammaL_grid)
    #Nbuf=1
    k = jnp.fft.rfftfreq(Ng_nu,dnu)
    valbuf=val
    #Nbuf=2
    #k = jnp.fft.rfftfreq(2*Ng_nu,dnu)
    #valbuf=jnp.vstack([val,jnp.zeros_like(val)])
    #Nbuf=4
    #k = jnp.fft.rfftfreq(4*Ng_nu,dnu)
    #valbuf=jnp.vstack([val,jnp.zeros_like(val),jnp.zeros_like(val),jnp.zeros_like(val)])

    fftval = jnp.fft.rfft(valbuf,axis=0)
    vk=voigt_kernel(k, beta_grid,gammaL_grid)
    fftvalsum = jnp.sum(fftval*vk,axis=(1,2))
    xs=jnp.fft.irfft(fftvalsum)[:Ng_nu]/dnu

    
    if True:
        km = jnp.fft.rfftfreq(2*Ng_nu,1)
#        km=k*dnu
        dv=dnu
        w=gammaL_grid #Ngw
        vmax=dv*Ng_nu/2.0 #N_v*dv 
        x=jnp.log(w/vmax) #Ngw
        A = w*jnp.exp(-(0.23299924*jnp.exp(x/0.53549119) + 6.74408847)) #Ngw
        B = w*jnp.exp(-(0.09226203*jnp.exp(x/0.49589094) + 6.82193751)) #Ngw
        w_corr = vmax*jnp.exp(0.18724358*jnp.exp(x/0.50806536) - 0.93309186) #Ngw
        w = w_corr/dv #Ngw
        gE_FT = 2.*w[None,:]/(1. + 4.*jnp.pi**2*km[:,None]**2*w[None,:]**2)  #Nnu x Ngw


        Err_corr = A[None,:]*gE_FT*100/vmax**2 #Nnu x Ngw

        #   later   
        zeroindex=jnp.zeros(len(k),dtype=int) #0 [Nnu]
        zeroindex=index_add(zeroindex, 0, 1.0)
        Err_corr = Err_corr + zeroindex[:,None]*B[None,:]*200/vmax/dv
        I_alternate = 1-(jnp.arange(len(k))&1)*2 #Nnu
        Err_corr = Err_corr*I_alternate[:,None]
        Err_corr=Err_corr[:Ng_nu]

    
#    print(jnp.shape(Err_corr)) #5001, 30
#    import sys
#    sys.exit()
    I_g_FT=  (fftval*vk)
    I_g_FT=I_g_FT[:Ng_nu] - Err_corr[:,None,:]/nu_grid[:,None,None]
#    I_g_FT = I_g_FT - (Err_corr[:,None,:])#/(nu_grid[:,None,None])
    xs = jnp.sum(jnp.fft.irfft(I_g_FT,axis=0),axis=(1,2))[:Ng_nu]/dnu
    return xs



@jit
def folded_voigt_kernel(k,beta,gammaL,Nfold,dnu, dLarray):
    """Folded Fourier Kernel of the Voigt Profile
    
    Args:
        k: conjugate wavenumber
        beta: Gaussian standard deviation
        gammaL: Lorentian Half Width
        Nfold: Folding number
        dnu: linear waveunmber grid size
        dLarray: dLarray

    Returns:
        kernel (N_x,N_beta,N_gammaL)
    
    Note:
        Conversions to the (full) width, wG and wL are as follows: 
        wG=2*sqrt(2*ln2) beta
        wL=gamma
    
    """

    def ffold(val,dL):
        val=val+jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*(k[:,None,None]+dL))**2 \
                              + 4.0*jnp.pi*gammaL[None,None,:]*(k[:,None,None]+dL)))
        val=val+jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*(k[:,None,None]-dL))**2 \
                              + 4.0*jnp.pi*gammaL[None,None,:]*(dL-k[:,None,None])))
        null=0.0
        return val, null
    
    val=jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*k[:,None,None])**2 + 4.0*jnp.pi*gammaL[None,None,:]*k[:,None,None]))
    
    val,nullstack=scan(ffold, val, dLarray)
    
    return val
    
@jit
def runditfold(S,nu_lines,beta,gammaL,nu_grid,beta_grid,gammaL_grid,Nfold,dLarray):
    """run DIT folded voigt

    Args:
       S: line strength (Nlines)
       nu_lines: line center (Nlines)
       beta: Gaussian STD (Nlines)
       gammaL: Lorentian half width (Nlines)
       nu_grid: linear wavenumber grid
       beta_grid: beta grid
       gammaL_grid: gammaL grid
       Nfold: # of folding
       dLarray: dLarray

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
    val=inc3D(S,nu_lines,log_beta,log_gammaL,nu_grid,log_beta_grid,log_gammaL_grid)
    #Nbuf=1
    k = jnp.fft.rfftfreq(Ng_nu,dnu)
    valbuf=val
    #Nbuf=2
    #k = jnp.fft.rfftfreq(2*Ng_nu,dnu)
    #valbuf=jnp.vstack([val,jnp.zeros_like(val)])
    #Nbuf=4
    #k = jnp.fft.rfftfreq(4*Ng_nu,dnu)
    #valbuf=jnp.vstack([val,jnp.zeros_like(val),jnp.zeros_like(val),jnp.zeros_like(val)])
    fftval = jnp.fft.rfft(valbuf,axis=0)
    vk=folded_voigt_kernel(k, beta_grid,gammaL_grid, Nfold, dnu, dLarray)
    fftvalsum = jnp.sum(fftval*vk,axis=(1,2))
    xs=jnp.fft.irfft(fftvalsum)[:Ng_nu]/dnu
    return xs

@jit
def f1_voigt_kernel(k,beta,gammaL,dnu):
    """Folded Fourier Kernel of the Voigt Profile for Nfold=1 (not using scan)
  
  
    Args:
        k: conjugate wavenumber
        beta: Gaussian standard deviation
        gammaL: Lorentian Half Width
        dnu: linear waveunmber grid size
        
    Returns:
        kernel (N_x,N_beta,N_gammaL)
    
    Note:
        This function is the folded voigt kernel but Nfold=1 without lax.scan
        Conversions to the (full) width, wG and wL are as follows: 
        wG=2*sqrt(2*ln2) beta
        wL=2*gamma
    
    """
    
    dL=1.0/dnu
    val=jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*k[:,None,None])**2 + jnp.pi*gammaL[None,None,:]*k[:,None,None]))
    val=val+jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*(k[:,None,None]+dL))**2 \
                              + jnp.pi*gammaL[None,None,:]*(k[:,None,None]+dL)))
    val=val+jnp.exp(-2.0*((jnp.pi*beta[None,:,None]*(k[:,None,None]-dL))**2 \
                              + jnp.pi*gammaL[None,None,:]*(dL-k[:,None,None])))   
    
    return val

def runditf1(S,nu_lines,beta,gammaL,nu_grid,beta_grid,gammaL_grid):
    """run DIT folded voigt but Nfold=1 (not using scan)

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
    vk=f1_voigt_kernel(k, beta_grid,gammaL_grid,dnu)
    fftvalsum = jnp.sum(fftval*vk,axis=(1,2))
    xs=jnp.fft.irfft(fftvalsum)[:Ng_nu]/dnu
    return xs

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
    
    xs=rundit(S,nu_lines,beta,gammaL,nus,beta_grid,gammaL_grid)
    xsf=runditfold(S,nu_lines,beta,gammaL,nus,beta_grid,gammaL_grid,2)
    xsf1=runditf1(S,nu_lines,beta,gammaL,nus,beta_grid,gammaL_grid)

    print("-beta-------------")
    print(beta)
    print(beta_grid)
    print("-gamma-------------")
    print(gammaL)
    print(gammaL_grid)

    xsv=xsection(nus,nu_lines,beta,gammaL,S)

    fig=plt.figure()
    fig.add_subplot(211)
    plt.plot(nus,xsv,label="exojax")
    plt.plot(nus,xs,label="addit")
    plt.plot(nus,xsf,label="addit f2")
    plt.plot(nus,xsf1,label="addit f1")

    plt.legend()
    for i in range(0,1):
        plt.axvline(nu_lines[i])
        plt.text(nu_lines[i],10.0,str(i))
    fig.add_subplot(212)
    plt.plot(nus,(xsv-xs),label="exojax-addit")
    plt.show()
