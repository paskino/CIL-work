#%%
import numpy as np
from cil.framework import ImageGeometry, AcquisitionGeometry, DataOrder
from cil.utilities.display import show2D, show_geometry, show1D
from cil.framework import AcquisitionData
from cil.recon import FDK
from cil.optimisation.functions import TotalVariation, L2NormSquared, WeightedL2NormSquared
from cil.plugins.tigre import ProjectionOperator
from cil.optimisation.algorithms import SPDHG, FISTA
from cil.optimisation.functions import BlockFunction
from cil.utilities.quality_measures import mse
from cil.framework import ImageData
import matplotlib.pyplot as plt
import json
import os
from cil.io import NEXUSDataWriter
import sys


#%%
# setup
if len(sys.argv) > 1:
     n_subs = int(sys.argv[1])
     gamma  = float(sys.argv[2])
     epochs = int(sys.argv[3])
     alpha  = float(sys.argv[4])
else:
     n_subs = 36
     gamma = 10
     epochs = 20
     alpha = 100

amin = 0
amax = 1.5

bbox = [150,230, 120,200]
vertical_slice = 170
horizontal_y = 180

dose = 'low'
filename = f'/opt/data/ICASSP24/train/train/0000_sino_{dose}_dose.npy'

data=np.asarray(np.load(filename,allow_pickle=True), dtype=np.float32)

#%%
print (data.shape)
print (DataOrder.ASTRA_AG_LABELS)
#%%
image_size = [300, 300, 300]
image_shape = [256, 256, 256]
voxel_size = [1.171875, 1.171875, 1.171875]

detector_shape = [256, 256]
detector_size = [600, 600]
pixel_size = [2.34375, 2.34375]

distance_source_origin = 575
distance_source_detector = 1050

#  S---------O------D
#

angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
#%%
AG = AcquisitionGeometry.create_Cone3D(source_position=[0, -distance_source_origin, 0],\
                                       detector_position=[0, distance_source_detector-distance_source_origin, 0],)\
                                        .set_angles(-angles, angle_unit='radian')\
                                        .set_panel(detector_shape, pixel_size, origin='bottom-left')\
                                        .set_labels(DataOrder.ASTRA_AG_LABELS[:])
ig = ImageGeometry(voxel_num_x=image_shape[0], voxel_num_y=image_shape[1], voxel_num_z=image_shape[2],\
                     voxel_size_x=voxel_size[0], voxel_size_y=voxel_size[1], voxel_size_z=voxel_size[2])
#%%
show_geometry(AG, ig)

# %%

ad = AcquisitionData(data, geometry=AG)

ad.reorder('tigre')
#%%

fdk = FDK(ad, ig).run()

# %%
show2D(fdk)
# %%


#%% 
# # Test in 2D
# data2d = ad.get_slice('centre')
# ig2d = ig.get_slice('centre')


#%%
g = alpha * TotalVariation(max_iteration=5)

#%%
#SPDHG
sdata = ad.partition(n_subs, 'staggered')
A = ProjectionOperator(ig, sdata.geometry)

f = [L2NormSquared(b=el) for el in sdata]
f = BlockFunction(*f)
# %%

spdhg = SPDHG(f=f, g=g, operator=A, max_iteration=100 * n_subs,\
      update_objective_interval=n_subs,\
      gamma=gamma)
#%%
spdhg.run(epochs * n_subs, verbose=2, print_interval=n_subs)
# %%

#%%
show2D([fdk, spdhg.solution], cmap=['gray', 'gray'])
# %%

plt.semilogy(spdhg.objective[1:])
plt.show()
# %%

gt = ImageData(np.load('/opt/data/ICASSP24/train/train/0000_clean_fdk_256.npy'),\
                         geometry=ig.copy())
# We will rank by the average MSE on the test set (unreleased)
dmse = mse(spdhg.solution, gt)
print (dmse)
# %%
# apply circular mask for display
savedir = os.path.join(os.getcwd(), 'results')

radius = 1
mfdk = fdk.apply_circular_mask(radius=radius, in_place=False)
malgo = spdhg.solution.apply_circular_mask(radius=radius, in_place=False)
mgt = gt.apply_circular_mask(radius=radius, in_place=False)

amax = max([mfdk.max(), malgo.max(), gt.max()])
amin = min([mfdk.min(), malgo.min(), gt.min()])
# amin = 0
# amax = 2
diff = mgt-malgo
lim = max(abs(diff.max()), abs(diff.min()))

# zoom in

show2D([el.get_slice(vertical=vertical_slice)\
               .as_array()[bbox[0]:bbox[1], bbox[2]:bbox[3]] \
          for el in [malgo, mfdk, mgt, diff]], \
     cmap=['viridis', 'viridis', 'viridis', 'seismic'],\
     title=[ f'SPDHG MSE {dmse}', 'FDK','Ground Truth', 'Difference'],\
     fix_range=[(amin, amax), (amin, amax), (amin, amax), (-lim/2, lim/2)])
# %%
show2D([el.get_slice(vertical=vertical_slice) \
     for el in [malgo, mfdk, mgt, diff]], \
     cmap=['viridis', 'viridis', 'viridis', 'seismic'],\
     title=[ f'SPDHG MSE {dmse}', 'FDK','Ground Truth', 'Difference'],\
     fix_range=[#(amin, amax), (amin, amax), (amin, amax),
     None, None, None, 
     (-lim/2, lim/2)])
# %%

show1D([el.get_slice(vertical=vertical_slice) \
     for el in [malgo, mfdk, mgt]], \
          label = ['SPDHG', 'FDK', 'Ground Truth'],\
          slice_list = [('horizontal_y', horizontal_y)], \
          title=f'Comparison of the reconstruction at y={horizontal_y}')

# %%
# save 

fname = os.path.join(savedir, f'SPDHG_{n_subs}_{gamma}_{alpha}_{epochs}.json')
fname2 = os.path.join(savedir, f'SPDHG_{n_subs}_{gamma}_{alpha}_{epochs}.nxs')


if not os.path.exists(savedir):
     os.makedirs(savedir)
     
w = NEXUSDataWriter()
w.set_up(data=spdhg.solution,
            file_name=fname2)
w.write()

res = {
     'n_subs': n_subs,
     'gamma': gamma,
     'epochs': epochs ,
     'alpha': alpha,
     'mse': dmse,
     'objective': spdhg.objective,
     'solution': fname2
}

with open(fname, 'w') as f:
     json.dump(res, f)
# %%
