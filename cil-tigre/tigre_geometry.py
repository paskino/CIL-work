
#%%from __future__ import division
import numpy as np
from tigre.utilities.geometry import Geometry
from tigre.utilities import Ax, Atb
from cil.framework import AcquisitionGeometry, ImageGeometry
from cil.optimisation.operators import LinearOperator
from tigre.algorithms import fdk

class CIL2TIGREGeometry(object):
    @staticmethod
    def getTIGREGeometry(ig, ag):
        tg = TIGREConeGeometry(ig, ag)
        tg.check_geo(angles=ag.config.angles.angle_data)
        return tg

class TIGREConeGeometry(Geometry):

    def __init__(self, ig, ag):

        Geometry.__init__(self)
        
        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        # ag.config.system.detector.position
        # ag.config.system.source.position
        # ag.config.system.rotation_axis.position
        # Distance Source Detector      (mm)
        self.DSD = ag.config.system.detector.position[1] - ag.config.system.source.position[1] 
        self.DSO = -ag.config.system.source.position[1]     # Distance Source Origin        (mm)
                                                            # Detector parameters
        # (V,U) number of pixels        (px)
        self.nDetector = np.array(ag.config.panel.num_pixels[::-1])
        # size of each pixel            (mm)
        self.dDetector = np.array(ag.config.panel.pixel_size[::-1])
        self.sDetector = self.dDetector * self.nDetector    # total size of the detector    (mm)
                                                            # Image parameters
        self.nVoxel = np.array( [1, ig.voxel_num_y, ig.voxel_num_x] )                   # number of voxels              (vx)
        self.dVoxel = np.array( [ig.voxel_size_x, ig.voxel_size_y, ig.voxel_size_x]  )                # size of each voxel            (mm)
        self.sVoxel = self.nVoxel * self.dVoxel             # total size of the image       (mm)
        
        # Offsets
        # self.offOrigin = np.array((0, 0, 0))                # Offset of image from origin   (mm)
        self.offOrigin = np.array( [0, 0, 0])
        self.offDetector = np.array( [0 , 0, 0])                 # Offset of Detector            (mm)
        self.rotDetector = np.array((0, 0, 0))
        # Auxiliary
        self.accuracy = 0.5                                 # Accuracy of FWD proj          (vx/sample)
        # Mode
        self.mode = 'cone'                                  # parallel, cone

# def Ax(img, geo, angles,  krylov="ray-voxel"):
class TIGREProjectionOperator(LinearOperator):
    def __init__(self, domain_geometry, range_geometry, method={'direct':'ray-voxel','adjoint': 'matched'}):
        super(TIGREProjectionOperator,self).__init__(domain_geometry=domain_geometry,\
             range_geometry=range_geometry)
        self.tigre_geom = CIL2TIGREGeometry.getTIGREGeometry(domain_geometry,range_geometry)

        # print (self.tigre_geom)

        self.angles = - range_geometry.config.angles.angle_data.copy()
        
        if range_geometry.config.angles.angle_unit == AcquisitionGeometry.DEGREE:
            self.angles *= (np.pi/180.) 
        
        self.method = method
    
    def direct(self, x, out=None):
        if out is None:
            out = self.range.allocate(None)
        
        print (x.shape, self.tigre_geom.nVoxel)
        data_temp = np.expand_dims(x.as_array(),axis=0)
        print (data_temp.shape)
        arr_out = Ax.Ax(data_temp, self.tigre_geom, self.angles , krylov=self.method['direct'])
        print (arr_out.shape)
        arr_out = np.squeeze(arr_out, axis=1)
        out.fill ( arr_out )
        return out
    def adjoint(self, x, out=None):
        if out is None:
            out = self.domain.allocate(None)
        data_temp = np.expand_dims(x.as_array(),axis=1)
        arr_out = Atb.Atb(data_temp, self.tigre_geom, self.angles , krylov=self.method['adjoint'])
        arr_out = np.squeeze(arr_out, axis=0)
        out.fill ( arr_out )
        return out
    def fdk(self, x):
        data_temp = np.expand_dims(x.as_array(),axis=1)
        arr_out = fdk(data_temp, self.tigre_geom, self.angles)
        arr_out = np.squeeze(arr_out, axis=0)
        return arr_out
        
#%%
if __name__ == '__main__':
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
    import os, sys
    

    #%% Read in data
    # path = "/media/scratch/Data/SophiaBeads/SophiaBeads_512_averaged/SophiaBeads_512_averaged.xtekct"


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
    # plotter2D([data_raw, ldata], cmap='gist_earth', stretch_y=True)
    #%%
    # ag = data.geometry
    # ig = ag.get_ImageGeometry()

    # data_cs = data.subset(vertical='centre')
    # ag_cs = data_cs.geometry

    ig_cs = ag_cs.get_ImageGeometry(resolution=1.0)
    #%%
    shift = shift_mm / ig_cs.voxel_size_x
    ag_shift = ag_cs.copy()
    ag_shift.config.system.rotation_axis.position= [shift,0.]

    # Try back/forward projection
    print ("Create TIGRE Projection Operator")
    A = TIGREProjectionOperator(domain_geometry=ig_cs, range_geometry=ag_cs, method={'direct':'ray-voxel', 'adjoint':'FDK'})
    print ("adjoint")
    bck = A.adjoint(ldata)
    rec = A.fdk(ldata)
    # plotter2D(bck, cmap='gist_earth')

    print ("direct")
    fwd = A.direct(bck)
    if fwd.mean() > 0:
        plotter2D([bck, fwd], cmap='gist_earth', stretch_y=True)
    sys.exit(0)
#%% Centre slice FDK
    
    fbp = FBP(ig_cs, ag_cs)
    fbp.set_input(ldata)
    FBP_cs = fbp.get_output()  
    #%%
    plotter2D([FBP_cs,rec], cmap='gist_earth',)
    sys.exit(0)
    data_cs = ldata
    # #%% Full 3D FDK
    # fbp = FBP(ig, ag)
    # fbp.set_input(data)
    # FBP_3D_out = fbp.get_output()  
    # plotter2D(FBP_3D_out.subset(vertical=999))
    

    #reconstruct the slice using FBP
    fbp = FBP(ig_cs, ag_shift)
    fbp.set_input(ldata)
    FBP_output = fbp.get_output()