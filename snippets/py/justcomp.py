from addit.dit import rundit
import jax.numpy as jnp
import numpy as np
from addit.ncf import inc3D
N=10000
Ng_nu=20000
Ng_beta=10
Ng_gammaL=10

nus=np.linspace(2050.0,2250.0,Ng_nu) #nu grid
beta_grid=jnp.logspace(-2,4,Ng_beta) #beta grid
gammaL_grid=jnp.logspace(-2,4,Ng_gammaL)#gammaL grid
S=np.logspace(0.0,2.0,N)
S[0:10]=1000.0

nu_lines=np.random.rand(N)*(nus[-1]-nus[0]-50.0)+nus[0]+25.0
beta=np.random.rand(N)*1.0
gammaL=np.random.rand(N)*1.0

jnu_lines=jnp.array(nu_lines)
jbeta=jnp.array(beta)
jgammaL=jnp.array(gammaL)

import time
ts=time.time()
#for i in range(0,1840):
for i in range(0,1198):
    F0in=rundit(S,jnu_lines,jbeta,jgammaL,nus,beta_grid,gammaL_grid)
te=time.time()
print(te-ts,"sec")
