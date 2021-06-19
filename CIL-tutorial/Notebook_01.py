#%%
import numpy as np
import os
import wget

from cil.io import TXRMDataReader, NEXUSDataReader
from cil.framework import AcquisitionGeometry, ImageGeometry
from cil.plugins.tigre import FBP as tFBP
from cil.plugins.astra.processors import FBP as aFBP
from cil.utilities.display import show2D
from cil.processors import CentreOfRotationCorrector
from cil.utilities.quality_measures import *

filename = os.path.abspath('/home/edo/scratch/Dataset/CCPi/valnut_tomo-A.txrm')
data_dir = os.getcwd()

# show2D(gt)
#%%
reader = TXRMDataReader()
angle_unit = AcquisitionGeometry.RADIAN

filename = os.path.abspath('/home/edo/scratch/Dataset/CCPi/valnut_tomo-A.txrm')

reader.set_up(file_name=filename, angle_unit=angle_unit)
data = reader.read()
if data.geometry is None:
    raise AssertionError("The reader should set the geometry!!")

# angles = data.geometry.angles * -1
# data.geometry.set_angles(angles)
#%%
show2D(data, slice_list=('vertical',600))
#%%
# get central slice
data2d = [ data.get_slice(vertical='centre') ] 
data2d.append(data2d[0].copy())

data2d[0].reorder(order='tigre')
data2d[1].reorder(order='astra')
#%%
# neg log
for d2d in data2d:
    if d2d.geometry is None:
        raise AssertionError('What? None?')
    d2d.log(out=d2d)
    d2d *= -1

#%%

ig2d = [ d2d.geometry.get_ImageGeometry() for d2d in data2d ]

#%%
angles = data.geometry.angles * -1.
data2d[0].geometry.set_angles(angles, angle_unit=AcquisitionGeometry.RADIAN)
data2d[1].geometry.set_angles(angles, angle_unit=AcquisitionGeometry.RADIAN)
fbp =  [ tFBP(ig2d[0],data2d[0].geometry) , aFBP(ig2d[1], data2d[1].geometry) ]
fbp[0].set_input(data2d[0])
fbp[1].set_input(data2d[1])
        
recon = [ fbp[0].get_output(), fbp[1].get_output() ]



show2D(recon, title=['TIGRE', 'ASTRA'])
# %%

# # # Construct the appropriate ImageGeometry
# ig2d = ImageGeometry(voxel_num_x=N,
#                     voxel_num_y=N,
#                     voxel_size_x=voxel_size_h, 
#                     voxel_size_y=voxel_size_h)
# if data2d.geometry is None:
#     raise AssertionError('What? None?')