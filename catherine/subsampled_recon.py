#%%
from cil.io import ZEISSDataReader, NEXUSDataReader
from cil.utilities.display import show2D, show1D
import os
import olefile
import dxchange
import numpy as np
from cil.framework import AcquisitionData
from tqdm import tqdm
from cil.processors import Slicer
from cil.plugins.astra import FBP
from cil.processors import Padder
from cil.processors import PaganinProcessor

#%%
data2d_abs = NEXUSDataReader("kidney_2D.nxs").read()


#%%
# paganin = PaganinProcessor()
# data2d_abs.geometry.config.units = 'cm'
# paganin.set_input(data2d_abs)
# data2d_abs = paganin.get_output()
# %% SET CENTRE OF ROTATION
cor = 4
data2d_abs.geometry.set_centre_of_rotation(4, distance_units='pixels')
ig = data2d_abs.geometry.get_ImageGeometry()


# %% TEST RECONSTRUCTION
data2d_abs.reorder('astra')
#%%
fbp = FBP(ig, data2d_abs.geometry, device="gpu")
fbp.set_input(data2d_abs)
recon_fbp = fbp.get_output()

show2D(recon_fbp, fix_range=(-0.05, 0.3))
# %% PAD DATA AS REGION OF INTEREST

pad = Padder.constant(pad_width={'horizontal': 300}, constant_values=0)
pad.set_input(data2d_abs)
data2d_abs_padded = pad.get_output()

#%% TEST RECONSTRUCTION
fbp = FBP(ig, data2d_abs_padded.geometry, device="gpu")
# %%
fbp.set_input(data2d_abs_padded)
recon_fbp = fbp.get_output()

show2D(recon_fbp, fix_range=(-0.05, 0.3))

# %% SIMULATE SUBSAMPLED ACQUISITION
reduction_factor = 4
slicer = Slicer(roi={'angle': (None, None, reduction_factor)})

slicer.set_input(data2d_abs_padded)

data_reduced = slicer.get_output()
#%% TEST RECONSTRUCTION
fbp = FBP(ig, data_reduced.geometry, device="gpu")

fbp.set_input(data_reduced)
recon_reduced = fbp.get_output()
# %%

# compare reduced and non-reduced reconstructions

show2D([el.apply_circular_mask(radius=0.9, in_place=False).array[300:700,700:1200] for el in [recon_fbp, recon_reduced]], fix_range=(-0.05, 0.25), title=['Full FBP', f'FBP reduction: {reduction_factor}'])
# %%


# Try some iterative stuff
from cil.optimisation.algorithms import APGD
from cil.optimisation.functions import L1Sparsity, LeastSquares
from cil.optimisation.operators import WaveletOperator
from cil.plugins.astra import ProjectionOperator

wavelet = WaveletOperator(ig, wavelet_name='haar', level=None)
l1 = L1Sparsity(wavelet)

A = ProjectionOperator(ig, data_reduced.geometry, device="gpu")
ls = LeastSquares(A, data_reduced, c=5e3)
#%%

apgd = APGD(initial=A.domain.allocate(0), f=ls, g=l1, update_objective_interval=10)

# %%
apgd.run(80)
# %%
show2D([el.apply_circular_mask(radius=0.9, in_place=False).array[350:750,850:1350] for el in [recon_fbp, recon_reduced, apgd.solution]], fix_range=(-0.05, 0.25), title=['Full FBP', f'FBP reduction: {reduction_factor}', 'LS+Wavelet'])
# %%
show1D([el.apply_circular_mask(radius=0.9, in_place=False).array[350:750,850:1350] for el in [recon_fbp, recon_reduced, apgd.solution]],
       dataset_labels=['Full FBP', f'FBP reduction: {reduction_factor}', 'LS+Wavelet'],
       slice_list=[(1, 100)])
# %%

import matplotlib.pyplot as plt
plt.semilogy(apgd.loss)
# %%
