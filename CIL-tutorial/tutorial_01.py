#%% Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cil.framework import ImageGeometry, AcquisitionGeometry
from cil.io import NikonDataReader, TIFFStackReader
# from cil.processors import CofR_FBP
from cil.plugins.astra.processors import FBP
import astra
import numpy as np

import matplotlib.pyplot as plt
from cil.utilities.jupyter import islicer
from cil.utilities.display import plotter2D
from cil.utilities.jupyter import link_islicer
import os

#%% Read in data
# path = "/media/scratch/Data/SophiaBeads/SophiaBeads_512_averaged/SophiaBeads_512_averaged.xtekct"

sparse_beads = False
if sparse_beads:
    path = "/mnt/data/CCPi/Dataset/SparseBeads/SparseBeads_ML_L3/CentreSlice"

    # Create a 2D fan beam Geometry

    source_position=(0, -121.9320936203)
    detector_position=(0, 1400.206 - source_position[1])
    num_projections = 2520
    angles = np.asarray([- 0.142857142857143 * i for i in range(num_projections)], dtype=np.float32)
    panel = 2000
    panel_pixel_size = 0.2

    ag_cs =  AcquisitionGeometry.create_Cone2D(source_position, detector_position)\
                                .set_angles(angles, angle_unit='degree')\
                                .set_panel(panel, pixel_size=panel_pixel_size, origin='top-right')

    #%%
    reader = TIFFStackReader()
    reader.set_up(file_name=os.path.join(path, 'Sinograms', 'SparseBeads_ML_L3_0001.tif'))
    data = reader.read_as_AcquisitionData(ag_cs)

    white_level = 60000.0

    data_raw = data.subset(dimensions=['angle','horizontal'])
    data_raw = data / white_level

    # negative log
    ldata = data_raw.log()
    ldata *= -1

    shift_mm = 12 * 0.2 / detector_position[1]
else:
    path = "/mnt/data/CCPi/Dataset/SophiaBeads_64_averaged/CentreSlice"

    # Create a 2D fan beam Geometry

    source_position=(0, -80.6392412185669)
    detector_position=(0, 1007.006 - source_position[1])
    angles = np.asarray([- 5.71428571428571 * i for i in range(63)], dtype=np.float32)
    panel = 2000
    panel_pixel_size = 0.2

    ag_cs =  AcquisitionGeometry.create_Cone2D(source_position, detector_position)\
                                .set_angles(angles, angle_unit='degree')\
                                .set_panel(panel, pixel_size=panel_pixel_size, origin='top-right')

    #%%
    reader = TIFFStackReader()
    reader.set_up(file_name=os.path.join(path, 'Sinograms', 'SophiaBeads_64_averaged_0001.tif'))
    data = reader.read_as_AcquisitionData(ag_cs)

    white_level = 60000.0

    data_raw = data.subset(dimensions=['angle','horizontal'])
    data_raw = data / white_level

    # negative log
    ldata = data_raw.log()
    ldata *= -1

    shift_mm = 0.0024




# data = data_raw.subset(dimensions=['vertical','angle','horizontal'])

#%% set up geometry
plotter2D([data_raw, ldata], cmap='gist_earth', stretch_y=True)
#%%
# ag = data.geometry
# ig = ag.get_ImageGeometry()

# data_cs = data.subset(vertical='centre')
# ag_cs = data_cs.geometry

ig_cs = ag_cs.get_ImageGeometry(resolution=1.0)

#%% Centre slice FDK

fbp = FBP(ig_cs, ag_cs)
fbp.set_input(ldata)
FBP_cs = fbp.get_output()  
#%%
plotter2D(FBP_cs, cmap='gist_earth',)
data_cs = ldata
# #%% Full 3D FDK
# fbp = FBP(ig, ag)
# fbp.set_input(data)
# FBP_3D_out = fbp.get_output()  
# plotter2D(FBP_3D_out.subset(vertical=999))
#%%
shift = shift_mm / ig_cs.voxel_size_x
ag_shift = ag_cs.copy()
ag_shift.config.system.rotation_axis.position= [shift,0.]

#reconstruct the slice using FBP
fbp = FBP(ig_cs, ag_shift)
fbp.set_input(ldata)
FBP_output = fbp.get_output()

#%%
plotter2D([FBP_cs, FBP_output],titles=['no centre', 'COR'], cmap='gist_earth',)

#%%
from cil.optimisation.functions import TotalVariation
from cil.optimisation.algorithms import FISTA
from cil.optimisation.functions import LeastSquares
from cil.plugins.astra.operators import ProjectionOperator as A#
from cil.plugins.ccpi_regularisation.functions import FGP_TV

K = A(ig_cs, ag_shift)
f = LeastSquares(K, ldata, c=0.5)
if sparse_beads:
    f.L = 1071.1967
else:
    f.L = 24.4184
alpha_rgl = 0.003
alpha = alpha_rgl * ig_cs.voxel_size_x
g = alpha * TotalVariation(lower=0.)
g = FGP_TV(alpha, 100, 1e-5, 1, 1, 0 , 'gpu')

algo = FISTA(initial=K.domain.allocate(0), f=f, g=g, max_iteration=10000, update_objective_interval=2)
#%%
algo.update_objective_interval=2
algo.run(10, verbose=1)

plotter2D(algo.solution, cmap='gist_earth')


# %%
from cil.optimisation.algorithms import PDHG
from cil.optimisation.functions import MixedL21Norm, BlockFunction, L2NormSquared, IndicatorBox
from cil.optimisation.operators import GradientOperator, BlockOperator

nabla = GradientOperator(ig_cs, backend='c')
F = BlockFunction(0.5 * L2NormSquared(b=ldata), alpha * MixedL21Norm())
BK = BlockOperator(K, nabla)
normK = BK.norm()

pdhg = PDHG(f=F, g=IndicatorBox(lower=0.), operator=BK, max_iteration=1000,
update_objective_interval=100)
#%%
# pdhg.run(1000, verbose=2)
#%%
plotter2D(pdhg.solution, cmap='gist_earth')


# %%

class AcquisitionGeometrySubsetGenerator(object):
    '''AcquisitionGeometrySubsetGenerator is a factory that helps generating subsets of AcquisitionData
    
    AcquisitionGeometrySubsetGenerator generates the indices to slice the data array in AcquisitionData along the 
    angle dimension with 4 methods:
    1. random: picks randomly between all the angles. Subset may contain same projection as other
    2. random_permutation: generates a number of subset by a random permutation of the indices, thereby no subset contain the same data.
    3. uniform: divides the angles in uniform subsets without permutation
    4. stagger: generates number_of_subsets by interleaving them, e.g. generating 2 subsets from [0,1,2,3] would lead to [0,2] and [1,3]
    The factory is not to be used directly rather from the AcquisitionGeometry class.
    '''
    
    ### Changes in the Operator required to work as OS operator
    @staticmethod
    def generate_subset(ag, subset_id, number_of_subsets, method='random'):
        
        angles = ag.angles.copy()
        if method == 'random':
            indices = [ AcquisitionGeometrySubsetGenerator.random_indices(angles, subset_id, number_of_subsets) 
              for _ in range(number_of_subsets) ] 
            
        elif method == 'random_permutation':
            rndidx = np.asarray(range(len(angles)))
            np.random.shuffle(rndidx)
            indices = AcquisitionGeometrySubsetGenerator.uniform_groups_indices(rndidx, number_of_subsets)
            
        elif method == 'uniform':
            rndidx = np.asarray(range(len(angles)))
            indices = AcquisitionGeometrySubsetGenerator.uniform_groups_indices(rndidx, number_of_subsets)
            
        elif method == 'stagger':
            idx = np.asarray(range(len(angles)))
            indices = AcquisitionGeometrySubsetGenerator.staggered_indices(idx, number_of_subsets)
        else:
            raise ValueError('Can only do {}. got {}'.format(['random', 'random_permutation', 'uniform'], method))
        return indices
    
    @staticmethod
    def uniform_groups_indices(idx, number_of_subsets):
        indices = []
        groups = int(len(idx)/number_of_subsets)
        for i in range(number_of_subsets):
            ret = np.asarray(np.zeros_like(idx), dtype=np.bool)
            for j,el in enumerate(idx[i*groups:(i+1)*groups]):
                ret[el] = True
                
            indices.append(ret)
        return indices
    @staticmethod
    def random_indices(angles, subset_id, number_of_subsets):
        N = int(np.floor(float(len(angles))/float(number_of_subsets)))
        indices = np.asarray(range(len(angles)))
        np.random.shuffle(indices)
        indices = indices[:N]
        ret = np.asarray(np.zeros_like(angles), dtype=np.bool)
        for i,el in enumerate(indices):
            ret[el] = True
        return ret
    @staticmethod
    def staggered_indices(idx, number_of_subsets):
        indices = []
        # groups = int(len(idx)/number_of_subsets)
        for i in range(number_of_subsets):
            ret = np.asarray(np.zeros_like(idx), dtype=np.bool)
            indices.append(ret)
        i = 0
        while i < len(idx):    
            for ret in indices:
                ret[i] = True
                i += 1
                if i >= len(idx):
                    break
                
        return indices
    @staticmethod
    def get_new_indices(index):
        newidx = []
        for idx in index:
            ai = np.where(idx == True)[0]
            for i in ai:
                newidx.append(i)
        return np.asarray(newidx)


subsets = AcquisitionGeometrySubsetGenerator.generate_subset(ag_shift, 0, 30, 'stagger')
print (subsets[0])
#%%
from cil.optimisation.algorithms import SPDHG
from cil.optimisation.functions import ZeroFunction
from cil.optimisation.functions import MixedL21Norm, BlockFunction, L2NormSquared, IndicatorBox
from cil.optimisation.operators import GradientOperator, BlockOperator


# fractionate the data
bldata = []
operators = []
funcs = []
for i , el in enumerate (subsets):
    tmpa = ldata.geometry.angles[el]
    geom = ag_shift.copy()
    geom.set_angles(tmpa, angle_unit=ldata.geometry.config.angles.angle_unit)
    tmp = geom.allocate(None)
    tmp.fill(ldata.as_array()[el])
    bldata.append(tmp)
    operators.append(A(ig_cs, bldata[-1].geometry))
    funcs.append(0.5*L2NormSquared(b=bldata[-1]))

operators.append(nabla)
funcs.append((alpha) * MixedL21Norm())

SBK = BlockOperator(*operators)
BF  = BlockFunction(*funcs)
ZF  = IndicatorBox(lower=0)

# normK = SBK.norm()
normK = 202.842
print (normK)
tau = np.asarray([1/normK for _ in range(len(SBK))])
sigma = np.asarray([1/normK for _ in range(len(SBK))])

spdhg = SPDHG(f=BF, g=ZF, operator=SBK,
    gamma=.5, max_iteration=pdhg.max_iteration*64,
    update_objective_interval=1000)

#%% 
spdhg.update_objective_interval = 100
spdhg.run(10000,verbose=2)


# %%
plotter2D(spdhg.solution, cmap='gist_earth')
# %%
