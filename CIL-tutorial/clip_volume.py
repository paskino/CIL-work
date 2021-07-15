import functools
from cil.io import NEXUSDataReader
writer = NEXUSDataReader()
writer.set_up(file_name='ovino.nxs')
recon = writer.read()
print (recon)

# show2D([data, recon], fix_range=(-0.01,0.06))
from ccpi.viewer import viewer2D, viewer3D
from ccpi.viewer.utils.conversion import Converter
import vtk
from functools import partial
import numpy as np

v  = viewer3D()

def update_clipping_plane(viewer, planew, interactor, event):
    # print ("Catching ", event)
    event_translator = planew.GetEventTranslator()
    pevent = event_translator.GetTranslation(event)
    print ("pevent", pevent)
    if pevent =='EndSelect' or pevent == 'Move':
        rep = planew.GetRepresentation()
        plane = vtk.vtkPlane()
        rep.GetPlane(plane)
        # print (plane)
    
        v.volume.GetMapper().RemoveAllClippingPlanes()
        v.volume.GetMapper().AddClippingPlane(plane)
        v.volume.Modified()
        v.getRenderer().Render()
    


def clipping_plane(v, interactor, event):
    if interactor.GetKeyCode() == "c":
        if hasattr(v, 'planew'):
            is_enabled = v.planew.GetEnabled()
            v.planew.SetEnabled(not is_enabled)
            print ("should set to not", is_enabled)
            v.getRenderer().Render()
        else:
            print ("handling c")
            planew = vtk.vtkImplicitPlaneWidget2()
            
            rep = vtk.vtkImplicitPlaneRepresentation()
            planew.SetInteractor(v.getInteractor())
            planew.SetRepresentation(rep)

            plane = vtk.vtkPlane()
            plane.SetOrigin(0,125,0)
            plane.SetNormal(0.,1.,0.)
            rep.GetPlane(plane)
            rep.UpdatePlacement()
            rep.PlaceWidget(v.volume.GetBounds())
            v.volume.GetMapper().AddClippingPlane(plane)
            v.volume.Modified()
            planew.On()
            v.plane = plane
            v.planew = planew
            v.style.AddObserver('LeftButtonReleaseEvent', functools.partial(update_clipping_plane, v, planew), 0.5)


v.style.AddObserver('KeyPressEvent', partial(clipping_plane, v), 0.5)

v.setInputData(Converter.numpy2vtkImage(recon.as_array()))
v.volume_colormap = 'inferno'


v.startRenderLoop()

print (v.plane.GetOrigin(), v.plane.GetNormal())
