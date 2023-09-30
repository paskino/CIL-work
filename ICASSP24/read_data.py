#%%
import numpy as np
from cil.framework import ImageGeometry, AcquisitionGeometry, DataOrder
from cil.utilities.display import show2D, show_geometry
from cil.framework import AcquisitionData
from cil.recon import FDK

#%%

dose = 'low'
filename = f'C:/Users/ofn77899/Data/ICASSP24/0000_sino_{dose}_dose.npy'

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

from cil.optimisation.functions import TotalVariation, L2NormSquared, IndicatorBox, LeastSquares
from cil.plugins.tigre import ProjectionOperator
from cil.optimisation.algorithms import SPDHG, FISTA

#%% 
# # Test in 2D
# data2d = ad.get_slice('centre')
# ig2d = ig.get_slice('centre')

#%%
A = ProjectionOperator(ig, AG)


F = LeastSquares(A, ad)
#%%
alpha = 100
g = alpha * TotalVariation(max_iteration=5)

#%%
algo = FISTA(fdk, F, g, max_iteration=100, update_objective_interval=5)
#%%
algo.run(10, verbose=1, print_interval=1)

#%%
show2D([fdk, algo.solution], cmap=['gray', 'gray'])
# %%
#SPDHG
from cil.optimisation.functions import BlockFunction
n_subs = 36
sdata = ad.partition(n_subs, 'staggered')
A = ProjectionOperator(ig, sdata.geometry)

f = [L2NormSquared(b=el) for el in sdata]
f = BlockFunction(*f)
# %%
gammas = [1e-4, 1e-2, 1e-1, 1e1, 1e4]

n_subs = 36

spdhg = SPDHG(f=f, g=g, operator=A, max_iteration=100, update_objective_interval=100,\
     gamma=gammas[1])
#%%
spdhg.run(20 * n_subs, verbose=2, print_interval=2)
# %%
show2D([fdk, spdhg.solution], cmap=['gray', 'gray'])
# %%
import matplotlib.pyplot as plt
plt.semilogy(spdhg.objective)
# %%
