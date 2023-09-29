#%%
import numpy as np
from cil.framework import ImageGeometry, AcquisitionGeometry, DataOrder
from cil.utilities.display import show2D, show_geometry
from cil.framework import AcquisitionData
from cil.recon import FDK

#%%

dose = 'low'
filename = f'C:/Users/ofn77899/Data/ICASSP24/0000_sino_{dose}_dose.npy'

data=np.load(filename,allow_pickle=True)

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
                                        .set_labels(DataOrder.ASTRA_AG_LABELS[1:])

#%%
show_geometry(AG)

# %%

ad = AcquisitionData(data, geometry=AG)

ad.reorder('tigre')
#%%

fdk = FDK(ad).run()

# %%
show2D(fdk)
# %%
