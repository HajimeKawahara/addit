from addit.dit import rundit
import jax.numpy as jnp
import numpy as np
from addit.ncf import inc3D
N=10000
Ng_nu=20000
Ng_beta=10
Ng_gammaL=10

nus=np.linspace(2050.0,2250.0,Ng_nu) #nu grid
beta_grid=np.logspace(-2,4,Ng_beta) #beta grid
gammaL_grid=np.logspace(-2,4,Ng_gammaL)#gammaL grid
S=np.logspace(0.0,2.0,N)
S[0:10]=1000.0

nu_lines=np.random.rand(N)*(nus[-1]-nus[0]-50.0)+nus[0]+25.0
beta=np.random.rand(N)*1.0
gammaL=np.random.rand(N)*1.0
F0in=rundit(S,nu_lines,beta,gammaL,nus,beta_grid,gammaL_grid)
F0obs=F0in+np.random.normal(size=len(nus))*np.median(F0in)*0.05 #5%


import arviz
import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi

def model_c(nu,y):
    sigma = numpyro.sample('sigma', dist.Exponential(np.median(F0in)*0.05))
    T = numpyro.sample('T', dist.Exponential(0.01))
    mu=rundit(S,nu_lines,beta*jnp.sqrt(T),gammaL,nus,beta_grid,gammaL_grid)
    
    numpyro.sample('y', dist.Normal(mu, sigma), obs=y)

from jax import random
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 1000, 2000

# In[17]:

kernel = NUTS(model_c)
mcmcx = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmcx.run(rng_key_, nu=nus, y=F0obs)
