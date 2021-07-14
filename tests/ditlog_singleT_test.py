"""This version is a test for a single beta (a single T) for DITlog.

"""


from addit.dit import rundit, runditfold, runditf1, make_dLarray
from addit.ditlog import rundit_fold_log,  rundit_fold_logred
from addit.ditlogst import rundit_fold_logredst
#rundit_fold_logredst

import jax.numpy as jnp
import numpy as np
from addit.ncf import inc3D
np.random.seed(20)

N=2000
Ng_nu=100000
Ng_gammaL=30

#log grid
nu0=2050.0
nu1=2150.0
nus=np.logspace(np.log10(nu0),np.log10(nu1),Ng_nu) #nu grid
R=(Ng_nu-1)/np.log(nu1/nu0) #resolution

avedv=np.mean(nus)/R #averaged d wavenumber
cnbeta=0.1/avedv #here we assume the normalized beta is constant


gammaL=np.random.rand(N)*1.0
#L grid
S=np.logspace(0.0,3.0,N)
S[0:20]=1.e5
gammaL[0:20]=0.01

nu_lines=np.random.rand(N)*(nus[-1]-nus[0]-50.0)+nus[0]+25.0
ngammaL_grid=np.logspace(np.log10(np.min(gammaL/nu_lines*R)),np.log10(np.max(gammaL/nu_lines*R)),Ng_gammaL)

#direct dv needs to be careful for the truncation error
nn=np.median(nus)

#dvx = jnp.interp(nu_lines-nn, nus[1:-1]-nn, 0.5*(nus[2:] - nus[:-2]))
dvx=nu_lines/R
ngammaL_grid_=np.logspace(np.log10(np.min(gammaL/dvx)),np.log10(np.max(gammaL/dvx)),Ng_gammaL)
dLarray=make_dLarray(2,1.0)
Nfold=1

#care for truncation
dv_lines=nu_lines/R
dv=nus/R
F0f2_=rundit_fold_logredst(S,nu_lines-nn,cnbeta,gammaL,nus-nn,ngammaL_grid_,dLarray,dv_lines,dv)

#direct voigt for comparison
import matplotlib.pyplot as plt
from exojax.spec import xsection
beta=cnbeta*dv_lines
xsv=xsection(nus,nu_lines,beta*np.ones_like(gammaL),gammaL,S)

fig=plt.figure()
ax=fig.add_subplot(211)
plt.plot(nus,xsv,label="direct")
plt.plot(nus,F0f2_,label="DIT (Nfold="+str(Nfold)+")",ls="dashed")
plt.plot(nus,F0f2_-xsv,label="DIT-direct (reduced $\nu$)")
#plt.xlim(2113.2,2113.5)

plt.legend()
ax=fig.add_subplot(212)
plt.plot(nus,(F0f2_-xsv),label="DIT/log (reduced $\nu$)",alpha=0.3)
plt.ylabel("difference")
#plt.xlim(2113.2,2113.5)
plt.legend()
plt.show()
