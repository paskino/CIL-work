from cil.optimisation.algorithms import FISTA, PDHG
from cil.optimisation.functions import LeastSquares, TotalVariation
from cil.optimisation.operators import BlurringOperator, ChannelwiseOperator
from cil.optimisation.utilities.callbacks import *
from cil.utilities import dataexample, noise
from cil.utilities.display import show2D

import numpy as np



#%%
data = dataexample.PEPPERS.get()


# Extract image geometry
ig = data.geometry

#%%
# Parameters for point spread function PSF (size and std)
ks          = 5; 
ksigma      = 2;

# Create 1D PSF and 2D as outer product, then normalise.
w           = np.exp(-np.arange(-(ks-1)/2,(ks-1)/2+1)**2/(2*ksigma**2))
w.shape     = (ks,1)
PSF         = w*np.transpose(w)
PSF         = PSF/(PSF**2).sum()
PSF         = PSF/PSF.sum()

# Display PSF as image
# show2D(PSF, origin="upper", title="PSF", size=(10,10))

# Create blurring operator and apply to clean image to produce blurred and display.
BOP = ChannelwiseOperator(BlurringOperator(PSF,ig), channels=3)

# blurred_noisy = noise.gaussian(BOP.direct(data), seed = 10, var = 0.0001)
blurred_noisy = BOP.direct(data)


# Show blurred and noisy image
show2D(blurred_noisy, origin="upper", title="Blurred+Noisy", size=(10,10))

#%%
# Setup and run FISTA algorithm 
alpha = 0.05
G = alpha * TotalVariation(max_iteration=3, warm_start=True)
F = LeastSquares(BOP, blurred_noisy)

fista = FISTA(initial = ig.allocate(0), f = F, g = G, 
              max_iteration = 200, update_objective_interval = 50)
fista.run(callbacks=[ProgressCallback(), TextProgressCallback()])  


#%%
show2D([data, blurred_noisy, fista.solution], 
       title=['Ground truth', 'Noisy Data (Gaussian)', 'Deblurred'], 
       origin="upper", num_cols=3)