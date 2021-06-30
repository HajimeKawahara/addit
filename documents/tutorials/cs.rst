Simple Usage
---------------

Let's assume that S,nu_lines,beta,gammaL are the arrays for line strength, line center, STD of a Gaussian, and a half width of Lorentian, with a dimension of # of the lines. First, set grids for DIT.

.. code:: ipython3
	  
    Ng_nu=20000
    Ng_beta=10
    Ng_gammaL=10

    nus=np.linspace(2050.0,2250.0,Ng_nu) #nu grid
    beta_grid=np.logspace(-2,4,Ng_beta) #beta grid
    gammaL_grid=np.logspace(-2,4,Ng_gammaL)#gammaL grid

and run DIT.
    
.. code:: ipython3
	      
    F0=rundit(S,nu_lines,beta,gammaL,nus,beta_grid,gammaL_grid)
