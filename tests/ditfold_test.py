from addit.dit import rundit, runditfold, runditnewfold, runditf1, make_dLarray
import jax.numpy as jnp
import numpy as np
from addit.ncf import inc3D
np.random.seed(20)

N=1
Ng_nu=10000
Ng_beta=29
Ng_gammaL=30

nus=np.linspace(2000.0,2250.0,Ng_nu) #nu grid
#vmax = 1000.0
#dv=0.1
#nus=np.arange(-vmax,vmax,dv)
#Ng_nu=len(nus)
#nus=np.linspace(-vmax,vmax,Ng_nu) #nu grid

beta=np.random.rand(N)*0.99+0.01
#gammaL=np.random.rand(N)*1.0
gammaL=np.ones(N)*10.0
beta_grid=np.logspace(np.log10(np.min(beta)),0,Ng_beta) #beta grid
gammaL_grid=np.logspace(np.log10(0.5),np.log10(2.5),Ng_gammaL)#gammaL grid
S=np.logspace(0.0,3.0,N)

nu_lines=np.random.rand(N)*(nus[-1]-nus[0]-200.0)+nus[0]+100.0

#F0=rundit(S,nu_lines,beta,gammaL,nus,beta_grid,gammaL_grid)
nn=np.median(nus)
dnu=nus[1]-nus[0]
dLarray=make_dLarray(2,dnu)

F0=rundit(S,nu_lines-nn,beta,gammaL,nus-nn,beta_grid,gammaL_grid)
F0fn=runditnewfold(S,nu_lines-nn,beta,gammaL,nus-nn,beta_grid,gammaL_grid)

F0f1=runditfold(S,nu_lines-nn,beta,gammaL,nus-nn,beta_grid,gammaL_grid,1,dLarray)

import matplotlib.pyplot as plt
from exojax.spec import xsection
xsv=xsection(nus,nu_lines,beta,gammaL,S)

fig=plt.figure()
ax=fig.add_subplot(211)
plt.plot(nus,xsv,label="direct")
plt.plot(nus,F0,label="DIT (0)")
plt.plot(nus,F0f1,label="DIT (fold)",ls="dashed")
plt.plot(nus,F0fn,label="DIT (new fold)",ls="dotted")
#plt.plot(nus,F0f1-xsv,label="DIT-direct")
plt.yscale("log")
plt.legend()
ax=fig.add_subplot(212)
plt.plot(nus,np.abs(F0/xsv-1),label="DIT-direct (0)",alpha=0.3)
plt.plot(nus,np.abs(F0f1/xsv-1),label="DIT-direct (fold)",alpha=0.3)
plt.plot(nus,np.abs(F0fn/xsv-1),label="DIT-direct (new fold)",alpha=0.3)
plt.ylim(1.e-4,100.0)
plt.yscale("log")
plt.legend()
plt.show()
