from addit.dit import rundit, runditfold, runditf1, make_dLarray
from addit.ditlog import rundit_fold_log
import jax.numpy as jnp
import numpy as np
from addit.ncf import inc3D
np.random.seed(20)

N=2000
Ng_nu=20000
Ng_beta=30
Ng_gammaL=30

#log grid
nu0=2050.0
nu1=2150.0
nus=np.logspace(np.log10(nu0),np.log10(nu1),Ng_nu) #nu grid
R=(Ng_nu-1)/np.log(nu1/nu0) #resolution

beta=np.random.rand(N)*0.99+0.01
gammaL=np.random.rand(N)*1.0
#L grid
S=np.logspace(0.0,3.0,N)
nu_lines=np.random.rand(N)*(nus[-1]-nus[0]-50.0)+nus[0]+25.0

nbeta_grid=np.logspace(np.log10(np.min(beta/nu_lines*R)),np.log10(np.max(beta/nu_lines*R)),Ng_beta) 
ngammaL_grid=np.logspace(np.log10(np.min(gammaL/nu_lines*R)),np.log10(np.max(gammaL/nu_lines*R)),Ng_gammaL)

dLarray=make_dLarray(2,1.0)
Nfold=1
F0f2=rundit_fold_log(S,nu_lines,beta,gammaL,nus,R,nbeta_grid,ngammaL_grid,dLarray)

import matplotlib.pyplot as plt
from exojax.spec import xsection
#direct
xsv=xsection(nus,nu_lines,beta,gammaL,S)

fig=plt.figure()
ax=fig.add_subplot(211)
plt.plot(nus,xsv,label="direct")
plt.plot(nus,F0f2,label="DIT (1)",ls="dashed")
plt.plot(nus,F0f2-xsv,label="DIT-direct")
plt.legend()
ax=fig.add_subplot(212)
plt.plot(nus,F0f2-xsv,label="DIT-log",alpha=0.3)
plt.legend()
plt.show()
