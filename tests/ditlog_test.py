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


dvc=nus/R
dvv = jnp.interp(nus, nus[1:-1], 0.5*(nus[2:] - nus[:-2]))

#import matplotlib.pyplot as plt
#plt.plot(nus,dvc-dvv,".")
#plt.show()
#import sys
#sys.exit()

beta=np.random.rand(N)*0.99+0.01
gammaL=np.random.rand(N)*1.0
#L grid
S=np.logspace(0.0,3.0,N)
S[0:20]=1.e5
beta[0:20]=0.01
gammaL[0:20]=0.01
nu_lines=np.random.rand(N)*(nus[-1]-nus[0]-50.0)+nus[0]+25.0

nbeta_grid=np.logspace(np.log10(np.min(beta/nu_lines*R)),np.log10(np.max(beta/nu_lines*R)),Ng_beta) 
ngammaL_grid=np.logspace(np.log10(np.min(gammaL/nu_lines*R)),np.log10(np.max(gammaL/nu_lines*R)),Ng_gammaL)


#direct dv needs to be careful for the truncation error
nn=np.median(nus)

#dvx = jnp.interp(nu_lines-nn, nus[1:-1]-nn, 0.5*(nus[2:] - nus[:-2]))
dvx=nu_lines/R
nbeta_grid_=np.logspace(np.log10(np.min(beta/dvx)),np.log10(np.max(beta/dvx)),Ng_beta) 
ngammaL_grid_=np.logspace(np.log10(np.min(gammaL/dvx)),np.log10(np.max(gammaL/dvx)),Ng_gammaL)


dLarray=make_dLarray(2,1.0)
Nfold=1
F0f2=rundit_fold_log(S,nu_lines,beta,gammaL,nus,R,nbeta_grid,ngammaL_grid,dLarray)

#care for truncation
dvx=nu_lines/R
dv=nus/R
F0f2_=rundit_fold_logred(S,nu_lines-nn,beta,gammaL,nus-nn,nbeta_grid_,ngammaL_grid_,dLarray,dvx,dv)

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
#plt.xlim(2113.2,2113.5)

plt.legend()
ax=fig.add_subplot(212)
plt.plot(nus,(F0f2-xsv),label="DIT/log",alpha=0.3)
plt.plot(nus,(F0f2_-xsv),label="DIT/log (reduced $\nu$)",alpha=0.3)
plt.ylabel("difference")
#plt.xlim(2113.2,2113.5)
plt.legend()
plt.show()
