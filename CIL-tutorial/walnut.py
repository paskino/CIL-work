import numpy as np
import os

from cil.io import TXRMDataReader, TIFFWriter
from cil.processors import TransmissionAbsorptionConverter, Binner
from cil.plugins.tigre import FBP
from cil.utilities.display import show2D, show_geometry

base_dir = os.path.abspath(r"C:\Users\ofn77899\Data\walnut")
data_name = "valnut"
filename = os.path.join(base_dir, data_name, "valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm")
is2D = False
data = TXRMDataReader(file_name=filename).read()
if is2D:
    data = data.get_slice(vertical='centre') 
else:
    binner = Binner(roi={'horizontal': (None, None, 4),'vertical': (None, None, 4)})
    data = binner(data)
    

# show_geometry(data.geometry)

data = TransmissionAbsorptionConverter()(data)

data.reorder(order='tigre')

ig = data.geometry.get_ImageGeometry()

fbp =  FBP(ig, data.geometry)
recon = fbp(data)
from cil.io import NEXUSDataWriter
writer = NEXUSDataWriter()
writer.set_up(data=recon, file_name=os.path.abspath('noce.nxs'))
writer.write()

# show2D([data, recon], fix_range=(-0.01,0.06))
from ccpi.viewer import viewer2D, viewer3D
from ccpi.viewer.utils.conversion import Converter
import vtk
from functools import partial

v  = viewer3D()

def clipping_plane(v, interactor, event):
    if interactor.GetKeyCode() == "c":
        planew = vtk.vtkImplicitPlaneWidget()
        planew.SetInteractor(v.getInteractor())
        plane = vtk.vtkPlane()
        planew.GetPlane(plane)
        planew.PlaceWidget()
        v.volume_mapper.AddClippingPlane(plane)
        v.plane = plane
        v.planew = planew

v.style.AddObserver('KeyPressEvent', partial(clipping_plane, v), 1.0)

v.setInputData(Converter.numpy2vtkImage(recon.as_array()))


v.startRenderLoop()

