import functools
# from cil.io import NEXUSDataReader


# show2D([data, recon], fix_range=(-0.01,0.06))
from ccpi.viewer import viewer2D, viewer3D
from ccpi.viewer.utils.conversion import Converter
import vtk
from functools import partial
import numpy as np
import os


def update_clipping_plane(viewer, planew, interactor, event):
    # print ("Catching ", event)
    event_translator = planew.GetEventTranslator()
    pevent = event_translator.GetTranslation(event)
    # print ("pevent", pevent)
    if pevent =='EndSelect' or pevent == 'Move' or pevent == 'NoEvent':
        rep = planew.GetRepresentation()
        plane = vtk.vtkPlane()
        rep.GetPlane(plane)
        # print (plane)
    
        viewer.volume.GetMapper().RemoveAllClippingPlanes()
        viewer.volume.GetMapper().AddClippingPlane(plane)
        viewer.volume.Modified()
        viewer.getRenderer().Render()
    


def clipping_plane(viewer, interactor, event):
    if interactor.GetKeyCode() == "c":
        if hasattr(viewer, 'planew'):
            is_enabled = viewer.planew.GetEnabled()
            viewer.planew.SetEnabled(not is_enabled)
            print ("should set to not", is_enabled)
            viewer.getRenderer().Render()
        else:
            print ("handling c")
            planew = vtk.vtkImplicitPlaneWidget2()
            
            rep = vtk.vtkImplicitPlaneRepresentation()
            planew.SetInteractor(viewer.getInteractor())
            planew.SetRepresentation(rep)

            plane = vtk.vtkPlane()
            plane.SetOrigin(0,125,0)
            plane.SetNormal(0.,1.,0.)
            rep.GetPlane(plane)
            rep.UpdatePlacement()
            rep.PlaceWidget(viewer.volume.GetBounds())
            viewer.volume.GetMapper().AddClippingPlane(plane)
            viewer.volume.Modified()
            planew.On()
            viewer.plane = plane
            viewer.planew = planew
            func = functools.partial(update_clipping_plane, viewer, planew)
            viewer.style.AddObserver('LeftButtonReleaseEvent', func, 0.5)
            planew.AddObserver('InteractionEvent', func, 0.5)

if __name__ == '__main__':
    
    v  = viewer3D()

    v.style.AddObserver('KeyPressEvent', partial(clipping_plane, v), 0.5)

    # writer = NEXUSDataReader()
    # writer.set_up(file_name='../CIL-tutorial/ovino.nxs')
    # recon = writer.read()
    # print (recon)
    # v.setInputData(Converter.numpy2vtkImage(recon.as_array()))
    
    reader = vtk.vtkNIFTIImageReader()
    data_dir = os.path.abspath('C:/Users/ofn77899/Data/PETMR/MCIR')
    fname  = 'gated_pdhg_Reg-FGP_TV-alpha3.0_nGates8_nSubsets1_pdhg_noPrecond_gamma1.0_wAC_wNorm_wRands-riters100_iters_5200.nii'

    data_dir = os.path.abspath('C:/Users/ofn77899/Downloads')
    fname = 'rec_im_mcir_fista_8ms_tgv.nii'

    fname = os.path.join(data_dir, fname)


    reader.SetFileName( fname )
    reader.Update()
    v.setInputData( reader.GetOutput() )
    v.volume_colormap_name = 'magma'


    v.startRenderLoop()

    print (v.plane.GetOrigin(), v.plane.GetNormal())
