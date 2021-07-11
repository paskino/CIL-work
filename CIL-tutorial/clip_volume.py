from cil.io import NEXUSDataReader
writer = NEXUSDataReader()
writer.set_up(file_name='noce.nxs')
recon = writer.read()
print (recon)
# show2D([data, recon], fix_range=(-0.01,0.06))
from ccpi.viewer import viewer2D, viewer3D
from ccpi.viewer.utils.conversion import Converter
import vtk
from functools import partial
import numpy as np

v  = viewer3D()

def clipping_plane(v, interactor, event):
    if interactor.GetKeyCode() == "c":
        print ("handling c")
        # planew = vtk.vtkImplicitPlaneWidget2()
        
        # rep = vtk.vtkImplicitPlaneRepresentation()
        # planew.SetInteractor(v.getInteractor())
        # planew.SetRepresentation(rep)

        plane = vtk.vtkPlane()
        plane.SetOrigin(0,125,0)
        plane.SetNormal(0.,1.,0.)
        # rep.GetPlane(plane)
        # rep.SetPlaceFactor(1.25)
        # rep.PlaceWidget(v.volume.GetBounds())
        v.volume.GetMapper().AddClippingPlane(plane)
        v.volume.Modified()
        # planew.On()
        v.plane = plane
        # v.planew = planew
        # planew.AddObserver('EndSelect', lambda: print (plane), 1.0)

v.style.AddObserver('KeyPressEvent', partial(clipping_plane, v), 1.0)

v.setInputData(Converter.numpy2vtkImage(recon.as_array()))
v.volume_colormap = 'inferno'


v.startRenderLoop()

print (v.plane.GetOrigin(), v.plane.GetNormal())
