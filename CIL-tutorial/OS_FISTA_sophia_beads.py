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

from ipywidgets import interact


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
    def generate_subset(ag, num_subsets, method='random'):
        
        angles = ag.angles.copy()
        if method == 'random':
            indices = [ AcquisitionGeometrySubsetGenerator.random_indices(angles, num_subsets) 
            for _ in range(num_subsets) ] 
            
        elif method == 'random_permutation':
            rndidx = np.asarray(range(len(angles)))
            np.random.shuffle(rndidx)
            indices = AcquisitionGeometrySubsetGenerator.uniform_groups_indices(rndidx, num_subsets)
            
        elif method == 'uniform':
            rndidx = np.asarray(range(len(angles)))
            indices = AcquisitionGeometrySubsetGenerator.uniform_groups_indices(rndidx, num_subsets)
            
        elif method == 'stagger':
            idx = np.asarray(range(len(angles)))
            indices = AcquisitionGeometrySubsetGenerator.staggered_indices(idx, num_subsets)
        else:
            raise ValueError('Can only do {}. got {}'.format(['random', 'random_permutation', 'uniform'], method))
        return indices
    
    @staticmethod
    def uniform_groups_indices(idx, num_subsets):
        indices = []
        groups = int(len(idx)/num_subsets)
        for i in range(num_subsets):
            ret = np.asarray(np.zeros_like(idx), dtype=np.bool)
            for j,el in enumerate(idx[i*groups:(i+1)*groups]):
                ret[el] = True
                
            indices.append(ret)
        return indices
    @staticmethod
    def random_indices(angles, num_subsets):
        N = int(np.floor(float(len(angles))/float(num_subsets)))
        indices = np.asarray(range(len(angles)))
        np.random.shuffle(indices)
        indices = indices[:N]
        ret = np.asarray(np.zeros_like(angles), dtype=np.bool)
        for i,el in enumerate(indices):
            ret[el] = True
        return ret
    @staticmethod
    def staggered_indices(idx, num_subsets):
        indices = []
        # groups = int(len(idx)/number_of_subsets)
        for i in range(num_subsets):
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



if __name__ == '__main__':
    path = os.path.abspath("/mnt/data/CCPi/Dataset/SophiaBeads_64_averaged/")

    # Create a 2D fan beam Geometry

    # source_position=(0, -80.6392412185669)
    # detector_position=(0, 1007.006 - source_position[1])
    # angles = np.asarray([- 5.71428571428571 * i for i in range(63)], dtype=np.float32)
    # panel = 2000
    # panel_pixel_size = 0.2

    # ag_cs =  AcquisitionGeometry.create_Cone2D(source_position, detector_position)\
    #                             .set_angles(angles, angle_unit='degree')\
    #                             .set_panel(panel, pixel_size=panel_pixel_size, origin='top-right')

    #%%
    # reader = TIFFStackReader()
    # reader.set_up(file_name=os.path.join(path, 'Sinograms', 'SophiaBeads_64_averaged_0001.tif'))
    # data = reader.read_as_AcquisitionData(ag_cs)
#%%
    # https://zenodo.org/record/16474#.YC9zWGj7Q10
    reader = NikonDataReader()
    reader.set_up(file_name=os.path.join(path, 'SophiaBeads_64_averaged.xtekct'))
    data = reader.read()
#%%
    white_level = 60000.0
    # ['vertical', 'angle', 'horizontal']
    # data_raw = data.subset(dimensions=['angle','horizontal'])
    data_raw = data.subset(dimensions=['vertical', 'angle', 'horizontal'])
    data_raw = data / white_level

    # negative log
    ldata = data_raw.log()
    del data_raw
    ldata *= -1

    shift_mm = 0.0024
    
    plotter2D(ldata.subset(vertical='centre'), cmap='gist_earth', stretch_y=True)
#%%
    # data = data_raw.subset(dimensions=['vertical','angle','horizontal'])

    # if jupyter is not available one could run 
    # somef('Sparse Beads')
    #%% set up geometry
    plotter2D([raw_data, ldata], cmap='gist_earth', stretch_y=True)
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
    # the c parameter is used to remove scaling of L2NormSquared in PDHG
    #
    c = 2
    f = LeastSquares(K, ldata, c=0.5 * c)
    if sparse_beads:
        f.L = 1071.1967 * c
    else:
        f.L = 24.4184 * c
    alpha_rgl = 0.003
    alpha = alpha_rgl * ig_cs.voxel_size_x
    g = c * alpha * TotalVariation(lower=0.)
    g = FGP_TV(alpha, 100, 1e-5, 1, 1, 0 , 'gpu')

    algo = FISTA(initial=K.domain.allocate(0), f=f, g=g, max_iteration=10000, update_objective_interval=2)
    #%%
    import cProfile
    algo.update_objective_interval=10
    cProfile.run('algo.run(100, verbose=1)')
    #%%
    plotter2D(algo.solution, cmap='gist_earth')

    #%%

    cProfile.run('algo.run(1)')

    # %%
    from cil.optimisation.algorithms import PDHG
    from cil.optimisation.functions import MixedL21Norm, BlockFunction, L2NormSquared, IndicatorBox
    from cil.optimisation.operators import GradientOperator, BlockOperator

    nabla = GradientOperator(ig_cs, backend='c')
    F = BlockFunction(L2NormSquared(b=ldata), alpha * MixedL21Norm())
    BK = BlockOperator(K, nabla)
    # normK = BK.norm()
    normK = 191.54791313753265

    pdhg = PDHG(f=F, g=IndicatorBox(lower=0.), operator=BK, max_iteration=1000,
    update_objective_interval=100)
    #%%
    pdhg.run(100, verbose=2, print_interval=10)
    #%%
    plotter2D(pdhg.solution, cmap='gist_earth')


    # %%


    subsets = AcquisitionGeometrySubsetGenerator.generate_subset(
        ag_shift, num_subsets=8, method='stagger'
        )
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
        update_objective_interval=1000, 
        # tau=tau, sigma=sigma
        )

    #%% 
    spdhg.update_objective_interval = 100
    spdhg.run(100,verbose=2)


    # %%
    plotter2D(spdhg.solution, cmap='gist_earth')
    # %%
    from ipywidgets import interact
    @interact(choice=['apples','oranges'])
    def somef(choice):
        print ("Chosen", choice)
    # interact(somef, choice=['apples','oranges'])

    # %%
    # import astra
    # from astra import astra_dict, algorithm, data3d
    # import numpy as np
    # def create_backprojection3d_gpu(data, proj_geom, vol_geom, returnData=True, vol_id=None):
    #     """Create a backprojection of a sinogram (3D) using CUDA.
    #     :param data: Sinogram data or ID.
    #     :type data: :class:`numpy.ndarray` or :class:`int`
    #     :param proj_geom: Projection geometry.
    #     :type proj_geom: :class:`dict`
    #     :param vol_geom: Volume geometry.
    #     :type vol_geom: :class:`dict`
    #     :param returnData: If False, only return the ID of the backprojection.
    #     :type returnData: :class:`bool`
    #     :returns: :class:`int` or (:class:`int`, :class:`numpy.ndarray`) -- If ``returnData=False``, returns the ID of the backprojection. Otherwise, returns a tuple containing the ID of the backprojection and the backprojection itself, in that order.
    # """
    #     if isinstance(data, np.ndarray):
    #         sino_id = data3d.create('-sino', proj_geom, data)
    #     else:
    #         sino_id = data

    #     if vol_id is None:
    #         vol_id = data3d.create('-vol', vol_geom, 0)

    #     cfg = astra_dict('BP3D_CUDA')
    #     cfg['ProjectionDataId'] = sino_id
    #     cfg['ReconstructionDataId'] = vol_id
    #     alg_id = algorithm.create(cfg)
    #     algorithm.run(alg_id)
    #     algorithm.delete(alg_id)

    #     if isinstance(data, np.ndarray):
    #         data3d.delete(sino_id)

    #     if vol_id is not None:
    #         if returnData:
    #             return vol_id, data3d.get_shared(vol_id)
    #         else:
    #             return vol_id
    #     else:
    #         if returnData:
    #             return vol_id, data3d.get(vol_id)
    #         else:
    #             return vol_id

    # def create_sino3d_gpu(data, proj_geom, vol_geom, returnData=True, gpuIndex=None, sino_id=None):
    #     """Create a forward projection of an image (3D).
    # :param data: Image data or ID.
    # :type data: :class:`numpy.ndarray` or :class:`int`
    # :param proj_geom: Projection geometry.
    # :type proj_geom: :class:`dict`
    # :param vol_geom: Volume geometry.
    # :type vol_geom: :class:`dict`
    # :param returnData: If False, only return the ID of the forward projection.
    # :type returnData: :class:`bool`
    # :param gpuIndex: Optional GPU index.
    # :type gpuIndex: :class:`int`
    # :returns: :class:`int` or (:class:`int`, :class:`numpy.ndarray`) -- If ``returnData=False``, returns the ID of the forward projection. Otherwise, returns a tuple containing the ID of the forward projection and the forward projection itself, in that order.
    # """

    #     if isinstance(data, np.ndarray):
    #         volume_id = data3d.create('-vol', vol_geom, data)
    #     else:
    #         volume_id = data
    #     if sino_id is None:
    #         sino_id = data3d.create('-sino', proj_geom, 0)

    #     algString = 'FP3D_CUDA'
    #     cfg = astra_dict(algString)
    #     if not gpuIndex==None:
    #         cfg['option']={'GPUindex':gpuIndex}
    #     cfg['ProjectionDataId'] = sino_id
    #     cfg['VolumeDataId'] = volume_id
    #     alg_id = algorithm.create(cfg)
    #     algorithm.run(alg_id)
    #     algorithm.delete(alg_id)

    #     if isinstance(data, np.ndarray):
    #         data3d.delete(volume_id)
    #     if sino_id is None:
    #         if returnData:
    #             return sino_id, data3d.get(sino_id)
    #         else:
    #             return sino_id
    #     else:
    #         if returnData:
    #             return sino_id, data3d.get_shared(sino_id)
    #         else:
    #             return sino_id


    # # %%
    # from cil.plugins.astra.utilities import convert_geometry_to_astra_vec

    # vol_geom, sino_geom = convert_geometry_to_astra_vec(ig_cs, ag_shift)

    # vol = ig_cs.allocate(None)
    # vol_id = astra.data3d.link('-vol', vol_geom, np.expand_dims(vol.as_array(), axis=0))

    # create_backprojection3d_gpu(np.expand_dims(ldata.as_array(), axis=0), sino_geom, vol_geom, False, vol_id)
    # sino = ag_shift.allocate(None)
    # sino_id = astra.data3d.link('-sino', sino_geom, np.expand_dims(sino.as_array(), axis=0))
    # create_sino3d_gpu(np.expand_dims(vol.as_array(), axis=0), sino_geom, vol_geom, False, None, sino_id)
    # plotter2D([vol, sino])
    # # %%
