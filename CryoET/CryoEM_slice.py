#%% imports
# import numpy as np
import mrcfile
from ccpi.io import *
from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData
from ccpi.optimisation.algorithms import CGLS, PDHG, FISTA
from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import L2NormSquared, ZeroFunction, MixedL21Norm, L1Norm, LeastSquares

from ccpi.astra.operators import AstraProjectorSimple , AstraProjector3DSimple
from ccpi.astra.processors import FBP
from ccpi.plugins.regularisers import FGP_TV
# from ccpi.utilities.jupyter import islicer, link_islicer
from ccpi.utilities.display import plotter2D

# All external imports
import numpy as np
import os
import sys
import scipy

import datetime
from PIL import Image

import functools
from ccpi.optimisation.operators import Identity
from ccpi.framework import BlockDataContainer
from ccpi.processors import Resizer

#%% Create TIFFWriter
class TIFFWriter(object):
    
    def __init__(self,
                 **kwargs):
        
        self.data_container = kwargs.get('data_container', None)
        self.file_name = kwargs.get('file_name', None)
        counter_offset = kwargs.get('counter_offset', 0)
        
        if ((self.data_container is not None) and (self.file_name is not None)):
            self.set_up(data_container = self.data_container,
                        file_name = self.file_name, 
                        counter_offset=counter_offset)
        
    def set_up(self,
               data_container = None,
               file_name = None,
               counter_offset = -1):
        
        self.data_container = data_container
        self.file_name = os.path.splitext(os.path.basename(file_name))[0]
        self.dir_name = os.path.dirname(file_name)
        self.counter_offset = counter_offset
        
        if not ((isinstance(self.data_container, ImageData)) or 
                (isinstance(self.data_container, AcquisitionData))):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')

    def write_file(self):
        '''alias of write'''
        return self.write()
    
    def write(self):
        if not os.path.isdir(self.dir_name):
            os.mkdir(self.dir_name)

        ndim = len(self.data_container.shape)
        if ndim == 2:
            # save single slice
            
            if self.counter_offset >= 0:
                fname = "{}_idx_{:04d}.tiff".format(os.path.join(self.dir_name, self.file_name), self.counter_offset)
            else:
                fname = "{}.tiff".format(os.path.join(self.dir_name, self.file_name))
            with open(fname, 'wb') as f:
                Image.fromarray(self.data_container.as_array()).save(f, 'tiff')
        elif ndim == 3:
            for sliceno in range(self.data_container.shape[0]):
                # save single slice
                # pattern = self.file_name.split('.')
                dimension = self.data_container.dimension_labels[0]
                fname = "{}_idx_{:04d}.tiff".format(
                    os.path.join(self.dir_name, self.file_name),
                    sliceno + self.counter_offset)
                with open(fname, 'wb') as f:
                    Image.fromarray(self.data_container.as_array()[sliceno]).save(f, 'tiff')
        elif ndim == 4:
            for sliceno1 in range(self.data_container.shape[0]):
                # save single slice
                # pattern = self.file_name.split('.')
                dimension = [ self.data_container.dimension_labels[0] ]
                for sliceno2 in range(self.data_container.shape[1]):
                    idx = self.data_container.shape[0] * sliceno2 + sliceno1 + self.counter_offset
                    fname = "{}_{}_{}_idx_{}.tiff".format(os.path.join(self.dir_name, self.file_name), 
                        self.data_container.shape[0], self.data_container.shape[1], idx)
                    with open(fname, 'wb') as f:
                        Image.fromarray(self.data_container.as_array()[sliceno1][sliceno2]).save(f, 'tiff')
        else:
            raise ValueError('Cannot handle more than 4 dimensions')


#%% Read in data

# set working directory
working_directory = os.path.abspath('/mnt/data/CCPi/Dataset/EM/Cryo_Sample_Data_1/')
os.chdir(working_directory)
# remove two angles for the aligned file
# angles should be in radians
angles = np.loadtxt('tomo_01.tlt') / 180. * np.pi

#read in full data set
with mrcfile.open('tomo_01.ali') as mrc:

    shape = mrc.data.shape
    vox_size = mrc.voxel_size
    
    ag_full = AcquisitionGeometry('parallel', '3D', 
                             pixel_num_h = shape[2],
                             pixel_num_v = shape[1],
                             pixel_size_h = np.floor(vox_size['y'] * 100. + 0.5)/100.,
                             pixel_size_v = np.floor(vox_size['z'] * 100. + 0.5)/100.,
                             angles = angles[:-2],
                             angle_unit='radian',
                             dimension_labels=['angle','vertical','horizontal'],
                            )

    data_raw = ag_full.allocate(None)
    data_raw.fill(mrc.data)

    # "normalise" in min/max range. Could be better to do in 1-99 percentile.
    data_raw.subtract(data_raw.min(), out=data_raw)
    data_raw.divide(data_raw.max()-data_raw.min(), out=data_raw)
    # transpose the data to match what Astra projector expects
    # take -log , remove zeros by adding epsilon
    epsilon = 1e-7
    data_full = -1 * (data_raw.subset(dimensions=['vertical','angle','horizontal'])+epsilon).log()
    ag_full = data_full.geometry
    min_value = data_full.min()
    del data_raw


#%% Set up processing
slices_output = 100
border_size = 0
slices_tot = slices_output + 2 * border_size
number_blocks = int(np.floor (ag_full.pixel_num_v / slices_output))

#set order to process blocks
#currently starting from the middle and working out
block_order = [None] * number_blocks
block_order[0] = int(np.floor(number_blocks/2))

sign = -1
for i in range(1,number_blocks):
	block_order[i] = int(block_order[i-1] + i * sign)
	sign *= -1

#%% run
for i in block_order:
    print("Processing block ", i)
    ind_start = i * slices_output
    ind_end = ind_start + slices_tot
        
    ag = ag_full.copy()
    ag.pixel_num_v = slices_tot

    data = ag.allocate(None)
    data.fill(data_full.as_array()[ind_start:ind_end,:,:])

    # stretch the reconstruction volume on x to be visible at maximum tilt angle in FOV
    # extend equally on both sides
    theta_max = np.max(np.abs(ag.angles))

    border_horizontal = int((ag.pixel_num_h / np.cos(theta_max) - ag.pixel_num_h) //2 + 1)
    
    num_vox_x = int(ag.pixel_num_h + 2 * border_horizontal)
    vox_size = ag.pixel_size_h

    # reconstruction volume
    ig = ImageGeometry( voxel_num_x=num_vox_x,
                        voxel_num_y=500,
                        voxel_num_z=ag.pixel_num_v,
                        voxel_size_x=vox_size,
                        voxel_size_y=vox_size,
                        voxel_size_z=vox_size)

    pad = True

    if pad:
        ag_pad = ag.copy()
        ag_pad.pixel_num_h = ig.voxel_num_x

        data_pad = ag_pad.allocate(None)
        
        m = border_horizontal
        M = m + ag.pixel_num_h

        data_pad.as_array()[:,:,m:M] = data.as_array()[:]
        for j in range(M, ig.voxel_num_x ):
            data_pad.as_array()[:,:,j] = data.as_array()[:,:,-1]
        for j in range(m):
            data_pad.as_array()[:,:,j] = data.as_array()[:,:,0]

        print ("The output image will be {:.3f} Gb".format( functools.reduce(lambda x,y: x*y, ig.shape, 1)/1024**3) )      
        

    if pad:
        Aop = AstraProjector3DSimple(ig, ag_pad)
        #data_block = BlockDataContainer(data_pad, zero_data)
        f = LeastSquares(A=Aop, b=data_pad)
        f.L = 58464.15624999999
    else:
        Aop = AstraProjector3DSimple(ig, ag)
        #data_block = BlockDataContainer(data, zero_data)
        f = LeastSquares(A=Aop, b=data)
    try:
        # algo = CGLS(operator=operator_block, data=data_block, 
        #                         update_objective_interval = 20, 
        #                         max_iteration = 1000)

        #%% Setup FISTA
        # r_alpha = 1
        # r_iterations = 100
        # r_tolerance = 1e-7          
        # r_iso = 1
        # r_nonneg = 0
        # r_printing = 0
        # TV = FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso, r_nonneg, r_printing, 'gpu')

        x_init = ig.allocate(0)
        algo = FISTA(x_init=x_init, f=f, g=ZeroFunction(), update_objective_interval = 20, max_iteration = 1000)
        #algo = FISTA(x_init=x_init, f=f, g=TV, update_objective_interval = 20, max_iteration = 1000)
        
        algo.run(100)

        roi = [(border_size,ig.voxel_num_z-border_size), -1, (border_horizontal,ig.voxel_num_x-border_horizontal)]       
        resizer = Resizer(roi=roi)
        resizer.set_input(algo.get_output())
        data_reconstructed = resizer.get_output()

        writer = TIFFWriter()
        writer.set_up(data_container=data_reconstructed, 
                    file_name="out3/LS_reco.tiff",
                    counter_offset=ind_start + border_size )
        writer.write()

        # def save_callback(iteration, obj, solution):
        #     # writer = TIFFWriter()
        #     # writer.set_up(data_container=solution, 
        #     #             file_name="./Block_CGLS_scale_1_gamma1_wide_it_{}.nxs".format(iteration)
        #     # )
        #     # writer.write_file()

        #     ig = ImageGeometry(voxel_num_x= int(num_vox_x / scale ),
        #             voxel_num_y= int(500. / scale ),# int(500. / scale), 
        #             voxel_num_z= ag.pixel_num_v, # int(ag.pixel_num_v/scale), #int(ag.pixel_num_v),
        #             voxel_size_x=vox_size,
        #             voxel_size_y=vox_size,
        #             voxel_size_z=vox_size)           
            
        #     diffx = num_vox_x - ag.pixel_num_h
            
        #     start = int(diffx/2)
        #     stop = int(start+ag.pixel_num_h)
            
        #     roi_crop = [(1,ig.voxel_num_z-1), -1, (start,stop)]
            
        #     resizer = Resizer(roi=roi_crop)
        #     resizer.set_input(solution)
        #     saveme = resizer.get_output()

        #     writer = TIFFWriter()
        #     writer.set_up(data_container=saveme, 
        #                 file_name="data_pad_overlap32/intel_ResizedRegularisedCGLS_Gradient_it_{:03d}.tiff".format(iteration),
        #                 counter_offset=i * saveme.geometry.voxel_num_z )
        #     writer.write()


        # algo.run(20, verbose=True, callback=save_callback)

    except MemoryError as me:
        print (me)


# %%
