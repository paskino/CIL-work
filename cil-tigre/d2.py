
import tigre
import numpy as np
from tigre import Ax
from tigre.demos.Test_data import data_loader
from cil.utilities.display import plotter2D

geo = tigre.geometry_default(high_quality=False)
print (geo.dDetector)
geo.nDetector[0] = 1
print (geo.nDetector)
geo.sDetector = geo.dDetector * geo.nDetector
print (geo.sDetector)
geo.nVoxel[0] = 1
geo.sVoxel = geo.nVoxel * geo.dVoxel
# print (geo)
# define angles
angles=np.linspace(0,2*np.pi,dtype=np.float32)
# load head phantom data
head=data_loader.load_head_phantom(number_of_voxels=geo.nVoxel)
# generate projections
projections=Ax(head,geo,angles,'interpolated')

print (geo.nVoxel, head.shape, projections.shape)
# plotter2D([head[0], projections[:,0,:]])