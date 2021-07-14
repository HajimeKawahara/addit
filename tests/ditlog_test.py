"""This version is a test for DITlog.

   * DITlog uses a 3D version of DIT
   * We need to be careful for the truncation error of the wavenumber grid.
   * To do so, reduced wavenumber is useful. This test compares both the original and reduced wavenumber grid and line center. 

"""

from addit.dit import rundit, runditfold, runditf1, make_dLarray
from addit.ditlog import rundit_fold_log,  rundit_fold_logred
import jax.numpy as jnp
import numpy as np
from addit.ncf import inc3D
np.random.seed(20)

N=2000
Ng_nu=100000
Ng_beta=30
Ng_gammaL=30

#log grid
nu0=2050.0
nu1=2150.0
nus=np.logspace(np.log10(nu0),np.log10(nu1),Ng_nu) #nu grid
R=(Ng_nu-1)/np.log(nu1/nu0) #resolution
beta=np.random.rand(N)*0.99+0.01
gammaL=np.random.rand(N)*1.0
S=np.logspace(0.0,3.0,N)
S[0:20]=1.e5
beta[0:20]=0.01
gammaL[0:20]=0.01
nu_lines=np.random.rand(N)*(nus[-1]-nus[0]-50.0)+nus[0]+25.0

nbeta_grid=np.logspace(np.log10(np.min(beta/nu_lines*R)),np.log10(np.max(beta/nu_lines*R)),Ng_beta) 
ngammaL_grid=np.logspace(np.log10(np.min(gammaL/nu_lines*R)),np.log10(np.max(gammaL/nu_lines*R)),Ng_gammaL)


#needs to be careful for the truncation error
nn=np.median(nus)
dv_lines=nu_lines/R
dv=nus/R
nbeta_grid_=np.logspace(np.log10(np.min(beta/dv_lines)),np.log10(np.max(beta/dv_lines)),Ng_beta) 
ngammaL_grid_=np.logspace(np.log10(np.min(gammaL/dv_lines)),np.log10(np.max(gammaL/dv_lines)),Ng_gammaL)
dLarray=make_dLarray(2,1.0)
Nfold=1


#using original wavenumber
F0f2=rundit_fold_log(S,nu_lines,beta,gammaL,nus,R,nbeta_grid,ngammaL_grid,dLarray)
#using reduced wavenumber
F0f2_=rundit_fold_logred(S,nu_lines-nn,beta,gammaL,nus-nn,nbeta_grid_,ngammaL_grid_,dLarray,dv_lines,dv)

#direct voigt for comparison
import matplotlib.pyplot as plt
from exojax.spec import xsection
xsv=xsection(nus,nu_lines,beta,gammaL,S)

fig=plt.figure()
ax=fig.add_subplot(211)
plt.plot(nus,xsv,label="direct")
plt.plot(nus,F0f2,label="DIT (Nfold="+str(Nfold)+")",ls="dashed")
plt.plot(nus,F0f2-xsv,label="DIT-direct")
plt.plot(nus,F0f2_-xsv,label="DIT-direct (reduced $\nu$)")
plt.legend(loc="upper right")
ax=fig.add_subplot(212)
plt.plot(nus,(F0f2-xsv),label="DIT/log",alpha=0.3)
plt.plot(nus,(F0f2_-xsv),label="DIT/log (reduced $\nu$)",alpha=0.3)
plt.ylabel("difference")
plt.legend(loc="upper right")
plt.show()
