from addit.dit import rundit, runditfold, runditf1, make_dLarray
import jax.numpy as jnp
import numpy as np
from addit.ncf import inc3D
np.random.seed(20)

N=2000
#Ng_nu=20000
Ng_nu=30000
Ng_beta=20
Ng_gammaL=20

nus=np.linspace(2050.0,2150.0,Ng_nu) #nu grid
beta=np.random.rand(N)*0.99+0.01
gammaL=np.random.rand(N)*1.0
beta_grid=np.logspace(np.log10(np.min(beta)),0,Ng_beta) #beta grid
gammaL_grid=np.logspace(np.log10(np.min(gammaL)),0,Ng_gammaL)#gammaL grid
S=np.logspace(0.0,3.0,N)

nu_lines=np.random.rand(N)*(nus[-1]-nus[0]-50.0)+nus[0]+25.0

F0=rundit(S,nu_lines,beta,gammaL,nus,beta_grid,gammaL_grid)

dnu=nus[1]-nus[0]
dLarray=make_dLarray(2,dnu)
F0=rundit(S,nu_lines,beta,gammaL,nus,beta_grid,gammaL_grid)
F0f1=runditfold(S,nu_lines,beta,gammaL,nus,beta_grid,gammaL_grid,1,dLarray)
F0f2=runditfold(S,nu_lines,beta,gammaL,nus,beta_grid,gammaL_grid,2,dLarray)

import matplotlib.pyplot as plt
from exojax.spec import xsection
xsv=xsection(nus,nu_lines,beta,gammaL,S)

fig=plt.figure()
ax=fig.add_subplot(211)
plt.plot(nus,xsv,label="direct")
plt.plot(nus,F0,label="DIT (0)",ls="dashed")
plt.plot(nus,F0f1,label="DIT (1)",ls="dashed")
plt.plot(nus,F0f2,label="DIT (2)",ls="dotted")
plt.plot(nus,F0f2-xsv,label="DIT-direct")
plt.legend()
ax=fig.add_subplot(212)
plt.plot(nus,F0-xsv,label="DIT-direct (0)",alpha=0.3)
plt.plot(nus,F0f1-xsv,label="DIT-direct (1)",alpha=0.3)
plt.plot(nus,F0f2-xsv,label="DIT-direct (2)",alpha=0.3)
plt.legend()
plt.show()
