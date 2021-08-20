from addit.dit import rundit, runditfold, rundit_newfold, runditf1, make_dLarray
import jax.numpy as jnp
import numpy as np
from addit.ncf import inc3D
from exojax.spec import xsection

np.random.seed(20)

N=1
Ng_nu=10000
Ng_beta=29
Ng_gammaL=30
tio=0.0
nus=np.linspace(2000.0-tio,2250.0+tio,Ng_nu) #nu grid
beta=np.random.rand(N)*0.99+0.01
gammaL=np.ones(N)*30.0
beta_grid=np.logspace(np.log10(np.min(beta)),0,Ng_beta) #beta grid
gammaL_grid=np.logspace(np.log10(0.3),np.log10(30.0),Ng_gammaL)#gammaL grid
S=np.logspace(0.0,3.0,N)
nu_lines=np.array([nus[0]+(nus[-1]-nus[0])/2.0])

#F0=rundit(S,nu_lines,beta,gammaL,nus,beta_grid,gammaL_grid)
nn=np.median(nus)
dnu=nus[1]-nus[0]
F0=rundit(S,nu_lines-nn,beta,gammaL,nus-nn,beta_grid,gammaL_grid)
F0_newf=rundit_newfold(S,nu_lines-nn,beta,gammaL,nus-nn,beta_grid,gammaL_grid)

xsv=xsection(nus,nu_lines,beta,gammaL,S)

gammaL=np.ones(N)*1.0
F0X=rundit(S,nu_lines-nn,beta,gammaL,nus-nn,beta_grid,gammaL_grid)
F0X_newf=rundit_newfold(S,nu_lines-nn,beta,gammaL,nus-nn,beta_grid,gammaL_grid)

xsvX=xsection(nus,nu_lines,beta,gammaL,S)

print(F0)
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot(211)
plt.plot(nus,xsv,label="direct",color="green",alpha=1.0)
plt.plot(nus,F0,label="DIT (0)",color="black",ls="dashed")
plt.plot(nus,F0_newf,label="DIT (new fold)",color="red",ls="dashed")

#plt.plot(nus,xsvX,label="direct",color="green",alpha=1.0)
#plt.plot(nus,F0X,label="DIT (0)",color="black",ls="dashed")
#plt.plot(nus,F0X_newf,label="DIT (new fold)",color="red",ls="dashed")

plt.plot(nus,xsvX,color="green",alpha=1.0)
plt.plot(nus,F0X,color="black",ls="dashed")
plt.plot(nus,F0X_newf,color="red",ls="dashed")
plt.title("LINEAR DIT")
plt.yscale("log")
plt.legend()

ax=fig.add_subplot(212)
plt.plot(nus,np.abs(F0/xsv-1),label="DIT-direct (0)",alpha=0.3,color="black")
plt.plot(nus,np.abs(F0X/xsvX-1),label="DIT-direct (0)",alpha=0.3,color="black")

plt.plot(nus,np.abs(F0_newf/xsv-1),label="DIT-direct (newfold)",alpha=0.3,color="red")
plt.plot(nus,np.abs(F0X_newf/xsvX-1),label="DIT-direct (newfold)",alpha=0.3,color="red")



plt.ylim(1.e-4,100.0)
plt.yscale("log")
plt.legend()
plt.savefig("newfold_linear.png")
plt.show()
