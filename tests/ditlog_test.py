"""This version is a test for DITlog.

   * DITlog uses a 3D version of DIT
   * We need to be careful for the truncation error of the wavenumber grid.
   * To do so, reduced wavenumber is useful. This test compares both the original and reduced wavenumber grid and line center. 

"""

from addit.dit import rundit, runditfold, runditf1, make_dLarray
from addit.ditlog import rundit_fold_log,  rundit_fold_logred, rundit_nofold_logred
import jax.numpy as jnp
import numpy as np
from addit.ncf import inc3D
from exojax.spec import xsection
np.random.seed(20)

N=1
Ng_nu=10000
Ng_beta=29
Ng_gammaL=30

nus=np.linspace(2000.0,2250.0,Ng_nu) #nu grid
beta=np.random.rand(N)*0.99+0.01
gammaL=np.ones(N)*30.0
beta_grid=np.logspace(np.log10(np.min(beta)),0,Ng_beta) #beta grid
gammaL_grid=np.logspace(np.log10(0.3),np.log10(30.0),Ng_gammaL)#gammaL grid
S=np.logspace(0.0,3.0,N)

nu_lines=np.random.rand(N)*(nus[-1]-nus[0]-200.0)+nus[0]+100.0


R=(Ng_nu-1)/np.log(nus[-1]/nus[0]) #resolution
nbeta_grid_=np.logspace(np.log10(np.min(beta/nu_lines*R)),np.log10(np.max(beta/nu_lines*R)),Ng_beta) 
ngammaL_grid_=np.logspace(np.log10(np.min(gammaL/nu_lines*R)),np.log10(np.max(gammaL/nu_lines*R)),Ng_gammaL)

#F0=rundit(S,nu_lines,beta,gammaL,nus,beta_grid,gammaL_grid)
nn=np.median(nus)
nn=np.median(nus)
dv_lines=nu_lines/R
dv=nus/R


dLarray=make_dLarray(2,1.0)
Nfold=1

#F0=rundit_fold_logred(S,nu_lines-nn,beta,gammaL,nus-nn,nbeta_grid_,ngammaL_grid_,dLarray,dv_lines,dv)
F0=rundit_nofold_logred(S,nu_lines-nn,beta,gammaL,nus-nn,nbeta_grid_,ngammaL_grid_,dv_lines,dv)

xsv=xsection(nus,nu_lines,beta,gammaL,S)

gammaL=np.ones(N)*1.0
ngammaL_grid_=np.logspace(np.log10(np.min(gammaL/nu_lines*R)),np.log10(np.max(gammaL/nu_lines*R)),Ng_gammaL)

#F0X=rundit_fold_logred(S,nu_lines-nn,beta,gammaL,nus-nn,nbeta_grid_,ngammaL_grid_,dLarray,dv_lines,dv)
F0X=rundit_nofold_logred(S,nu_lines-nn,beta,gammaL,nus-nn,nbeta_grid_,ngammaL_grid_,dv_lines,dv)

xsvX=xsection(nus,nu_lines,beta,gammaL,S)

print(F0)
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot(211)
plt.plot(nus,xsv,label="direct",color="C0",alpha=0.3)
plt.plot(nus,F0,label="DIT (0)",color="black",ls="dashed")

plt.plot(nus,xsvX,label="direct",color="C1",alpha=0.3)
plt.plot(nus,F0X,label="DIT (0)",color="black",ls="dashed")

#plt.plot(nus,F0fn,label="DIT (new fold)",ls="dotted")
#plt.plot(nus,F0f1-xsv,label="DIT-direct")
plt.yscale("log")
plt.legend()

ax=fig.add_subplot(212)
plt.plot(nus,np.abs(F0/xsv-1),label="DIT-direct (0)",alpha=0.3,color="C0")
plt.plot(nus,np.abs(F0X/xsvX-1),label="DIT-direct (0)",alpha=0.3,color="C1")

plt.ylim(1.e-4,100.0)
plt.yscale("log")
plt.legend()
plt.show()




