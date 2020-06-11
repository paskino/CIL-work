import numpy as np
import mrcfile
from ccpi.io import *
from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData
from ccpi.optimisation.algorithms import CGLS, PDHG, FISTA
from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import L2NormSquared, ZeroFunction, MixedL21Norm, L1Norm

from ccpi.astra.operators import AstraProjectorSimple , AstraProjector3DSimple
from ccpi.astra.processors import FBP

# from ccpi.utilities.jupyter import islicer, link_islicer
from ccpi.utilities.display import plotter2D

# All external imports
import numpy as np
import os
import sys
import scipy

import numpy as np
import os
from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
import datetime
from PIL import Image

import functools
from ccpi.optimisation.operators import Identity
from ccpi.framework import BlockDataContainer
from ccpi.processors import Resizer


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


# set working directory

working_directory = os.path.abspath('/mnt/data/CCPi/Dataset/EM/Cryo_Sample_Data_1/')
os.chdir(working_directory)
# remove two angles for the aligned file
# angles should be in radians
angles = np.loadtxt('tomo_01.tlt') / 180. * np.pi

# vertical band around a centre slice data to be used during fit
vertical_band = 50
# stride between consequent centre slices
stride = 3*vertical_band // 2
# reconstructed data discarded (equal to the overlap between vertical bands)
discard =  stride - vertical_band
vband = vertical_band

centre_slices = []
bands = []
offsets = []
with mrcfile.open('tomo_01.ali') as mrc:
    #ndarray = mrc.data.copy()
    shape = mrc.data.shape
    print ("MRC data shape", shape)
    vpanel_v_size = 2*vband+1 if vband != 0 else shape[1]
    for i,sl in enumerate(range(vband, shape[1], stride)):
        centre_slices.append(sl)
        max_band = centre_slices[-1]+vband
        bands.append([centre_slices[-1]-vband,max_band])
    # reconstruct on a smaller virtual panel with 5 slices per side
    

print (centre_slices)
print (bands)
print(offsets)
# # remove two angles for the aligned file
# # angles should be in radians
# angles = np.loadtxt('tomo_01.tlt') / 180. * np.pi


#for i in range(len(centre_slices)):
#for i in range(3, len(centre_slices)):
# j = len(centre_slices)//2
for i in [len(centre_slices)//5]:
    centre_slice = centre_slices[i]


    with mrcfile.open('tomo_01.ali') as mrc:
        #ndarray = mrc.data.copy()
        shape = mrc.data.shape
        print ("MRC data shape", shape)
        
        vox_size = mrc.voxel_size
        print (vox_size)
        
        ag = AcquisitionGeometry('parallel', '3D', 
                                pixel_num_h = shape[2],
                                pixel_num_v = vpanel_v_size, 
                                pixel_size_h = np.floor(np.float32(vox_size['y'])* 100 + 0.5 ) / 100.,
                                pixel_size_v = np.floor(np.float32(vox_size['z'])* 100 + 0.5 ) / 100.,
                                angles = angles[:-2],
                                angle_unit='radian',
                                dimension_labels=['angle', 'vertical', 'horizontal' ],
                                )
        tomo = ag.allocate(None)
        #tomo.fill(mrc.data[:, centre_slice - vband: centre_slice+vband+1, :])
        tomo.fill(mrc.data[:, bands[i][0]: bands[i][1]+1, :])
        # print (mrc.data.shape)
        
    # "normalise" in min/max range. Could be better to do in 1-99 percentile.
    tomo.subtract(tomo.min(), out=tomo)
    tomo.divide(tomo.max()-tomo.min(), out=tomo)
    # transpose the data to match what Astra projector expects
    # take -log , remove zeros by adding epsilon
    epsilon = 1e-7
    tomost = -1 * (tomo.subset(dimensions=['vertical','angle','horizontal'])+epsilon).log()
    del tomo

    scale = 1
    offset = 500. # was 500.
    vox_size = ag.pixel_size_h * scale

    # stretch the reconstruction volume on x to be visible at maximum tilt angle in FOV
    theta_max = np.max(np.abs(ag.angles))
    num_vox_x = ag.pixel_num_h / np.cos(theta_max)

    # because the reconstruction volume is not a cube try to maintain the same memory footprint
    # num_vox_y = ag.pixel_num_v

    # single slice
    ig = ImageGeometry(voxel_num_x= int(num_vox_x / scale ),
                    voxel_num_y= int(500. / scale ),# int(500. / scale), 
                    voxel_num_z= ag.pixel_num_v, # int(ag.pixel_num_v/scale), #int(ag.pixel_num_v),
                    voxel_size_x=vox_size,
                    voxel_size_y=vox_size,
                    voxel_size_z=vox_size)

    print (functools.reduce(lambda x,y: x*y, ig.shape, 1)/1024**3)
    print (ig)

    # pad acquisition Data
    pad = True

    if pad:
        ag = tomost.geometry
        print (ag.dimension_labels)
        ag_pad = AcquisitionGeometry('parallel', '3D', 
                    pixel_num_h = ig.voxel_num_x, 
                    pixel_num_v = ag.pixel_num_v, 
                    pixel_size_h=ag.pixel_size_h, 
                    pixel_size_v=ag.pixel_size_v, 
                    channels=1,
                    angles = ag.angles.copy(), 
                    dimension_labels=ag.dimension_labels)

        print (ag_pad)
        data_pad = ag_pad.allocate(tomost.min())
        diff = ig.voxel_num_x - tomost.geometry.pixel_num_h
        print ("DIFF",diff, diff/2)
        m = int(diff/2)
        M = m + tomost.geometry.pixel_num_h
        print ("m {} M {}".format(m,M))
        data_pad.as_array()[:,:,m:M] = tomost.as_array()[:]
        for j in range(M, ig.voxel_num_x ):
            data_pad.as_array()[:,:,j] = tomost.as_array()[:,:,-1]
        for j in range(m):
            data_pad.as_array()[:,:,j] = tomost.as_array()[:,:,0]

        print ("The output image will be {:.3f} Gb".format( functools.reduce(lambda x,y: x*y, ig.shape, 1)/1024**3) )
        print (ig)
        # setup the projector
        A = AstraProjector3DSimple(ig, ag_pad)
    else:
        A = AstraProjector3DSimple(ig, ag)
    L = Gradient(ig)
    # if scale == 5:
    #     alpha = 111.
    # if scale == 1:
    #     alpha = 50.281313
    # elif scale == 2:
    #     alpha = 72
    if scale == 1:
        alpha = 50.082964
    # if False:
    #   pass
    else:
        try:
            # alpha prop to ratio of norm of A and norm of L
            print ("calculating norm of A")
            normA = A.norm(verbose=True)
            print ("A.norm = ", normA)
            print ("calculating norm of Gradient")
            normL = L.norm(verbose=True)
            print ("L.norm = ", normL)
            ratio = normA/normL
            print ("calculating norm ratio", ratio)
        except MemoryError as me:
            print (me)
            raise ValueError('Out of memory')
        # gamma selects the weighing between the regularisation and the fitting term:
        # 1 means equal weight
        # 
        gamma = 1
        alpha = gamma * ratio
        print (alpha, normA, normL, normL/normA, normA/normL)


    # setup the Regularised CGLS 

    operator_block = BlockOperator( A, alpha * L)

    zero_data = L.range_geometry().allocate(0)

    if pad:
        data_block = BlockDataContainer(data_pad, zero_data)
    else:
        data_block = BlockDataContainer(tomost, zero_data)
    
    try:
        algo = CGLS(operator=operator_block, data=data_block, 
                                update_objective_interval = 20, 
                                max_iteration = 1000)
        

        def save_callback(iteration, obj, solution):
            # writer = TIFFWriter()
            # writer.set_up(data_container=solution, 
            #             file_name="./Block_CGLS_scale_1_gamma1_wide_it_{}.nxs".format(iteration)
            # )
            # writer.write_file()

            ig = ImageGeometry(voxel_num_x= int(num_vox_x / scale ),
                    voxel_num_y= int(500. / scale ),# int(500. / scale), 
                    voxel_num_z= ag.pixel_num_v, # int(ag.pixel_num_v/scale), #int(ag.pixel_num_v),
                    voxel_size_x=vox_size,
                    voxel_size_y=vox_size,
                    voxel_size_z=vox_size)

            
            
            diffx = num_vox_x - ag.pixel_num_h
            
            start = int(diffx/2)
            stop = int(start+ag.pixel_num_h)
            
            roi_crop = [(1,ig.voxel_num_z-1), -1, (start,stop)]
            
            resizer = Resizer(roi=roi_crop)
            resizer.set_input(solution)
            saveme = resizer.get_output()

            writer = TIFFWriter()
            writer.set_up(data_container=saveme, 
                        file_name="data_pad_overlap32/intel_ResizedRegularisedCGLS_Gradient_it_{:03d}.tiff".format(iteration),
                        counter_offset=i * saveme.geometry.voxel_num_z )
            writer.write()


        algo.run(80, verbose=True, callback=save_callback)
    except MemoryError as me:
        print (me)
