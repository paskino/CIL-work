from cil.framework import ImageGeometry, AcquisitionGeometry, DataOrder,\
    AcquisitionData
import numpy as np
import os


def create_ActisitionGeometry():
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
    
    return AG

def create_ImageGeometry():
    image_size = [300, 300, 300]
    image_shape = [256, 256, 256]
    voxel_size = [1.171875, 1.171875, 1.171875]

    ig = ImageGeometry(voxel_num_x=image_shape[0], 
                       voxel_num_y=image_shape[1],\
                       voxel_num_z=image_shape[2],\
                       voxel_size_x=voxel_size[0],\
                       voxel_size_y=voxel_size[1],\
                       voxel_size_z=voxel_size[2])

    return ig

def import_data(filename):
    data = np.asarray(
                np.load(filename,allow_pickle=True),\
            dtype=np.float32)

    AG = create_ActisitionGeometry()
    ad = AcquisitionData(data, geometry=AG)
    return ad

def get_AcquisitionData(base_dir, phantom=0, dose='low'):
    if dose not in ['low', 'clinical']:
        raise ValueError('dose must be low or clinical')
    fname = os.path.join(base_dir, f'{phantom:04d}_sino_{dose}_dose.npy')
    return import_data(fname)
#%%