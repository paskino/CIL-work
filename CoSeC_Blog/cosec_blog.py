import numpy as np
import os

from cil.io import TXRMDataReader, NEXUSDataWriter, TIFFWriter, NEXUSDataReader
from cil.processors import TransmissionAbsorptionConverter, Binner, CentreOfRotationCorrector, Slicer

from cil.utilities.display import show2D, show_geometry

from cil.recon import FDK
from cil.plugins.tigre import FBP

import os, sys



def reconstruct(filename):

    data = TXRMDataReader(file_name=filename).read()
    data.reorder('tigre')
    data = CentreOfRotationCorrector.image_sharpness(FBP=FBP)(data)

    binner = Binner(roi={'vertical':(None, None, 2), 'horizontal':(None, None,2)})
    data = binner(data)
    print(data.geometry)
    print(data.geometry.magnification)

    # Decide on the reconstruction volume
    ig = data.geometry.get_ImageGeometry(resolution=0.4)

    num_vox_x = int(ig.voxel_num_x * 0.88)
    num_vox_y = int(ig.voxel_num_y * 0.82)

    ig.voxel_num_x = num_vox_x
    ig.voxel_num_y = num_vox_y

    data = TransmissionAbsorptionConverter()(data)

    # run fdk
    fdk = FDK(data, ig)
    recon = fdk.run()

    # save some reconstructions and the data for later
    writer = NEXUSDataWriter(data=recon, file_name='ovino2.nxs')
    writer.write()
    writer = NEXUSDataWriter(data=recon, file_name='ovino2_8.nxs', compress=16)
    writer.write()
    writer = NEXUSDataWriter(data=data, file_name='data_ovino2.nxs')
    writer.write()

def save_radiographs(stride=20):
    adata = NEXUSDataReader(file_name='data_ovino2.nxs').read()
    reduced = Slicer(roi={'angle':(0, -1, stride)})(adata)

    writer = TIFFWriter(data=reduced, file_name='projections')
    writer.write()

if __name__ == '__main__':

    if len(sys.argv) == 1 or sys.argv[1] == '--recon':
        # install 
        base_dir = os.path.abspath("C:/Users/ofn77899/Data/egg2/gruppe2/gruppe2_2014-03-20_1105_07/tomo-A")
        filename = os.path.abspath(os.path.join(base_dir, "gruppe2_tomo-A.txrm"))

        reconstruct(filename)

    elif sys.argv[1] == '--save_proj':
        save_radiographs()

    elif sys.argv[1] == '--show_geometry':
        adata = NEXUSDataReader(file_name='data_ovino2.nxs').read()
        recon = NEXUSDataReader(file_name='ovino2_8.nxs').read()
        show_geometry(adata.geometry, recon.geometry)
    
