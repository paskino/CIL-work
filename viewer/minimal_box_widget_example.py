import vtk
import functools


import vtk

def create_sample_image(dims):
    # Create a black image
    img = vtk.vtkImageData()
    img.SetDimensions(dims)
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    img.GetPointData().GetScalars().Fill(0)

    return img

def add_sphere(img, radius, center):
    dims = img.GetDimensions()
    # Create a white circle
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if (i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2 < radius**2:
                    img.SetScalarComponentFromDouble(i, j, k, 0, 255)

    return 0

def add_cylinder(img, radius, center, dimz):
    dims = img.GetDimensions()
    # Create a white circle
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if (i - center[0])**2 + (j - center[1])**2 < radius**2 and dimz[0] < k < dimz[1]:
                    img.SetScalarComponentFromDouble(i, j, k, 0, 255)
    return 0


def add_parallelogram(img, dimx, dimy, dimz):
    dims = img.GetDimensions()
    # Create a white circle
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if dimx[0] < i < dimx[1] and dimy[0] < j < dimy[1] and dimz[0] < k < dimz[1]:
                    img.SetScalarComponentFromDouble(i, j, k, 0, 255)
    return 0

def observeCharEvent(img, box_widget, interactor, obj, event):
    print(event)
    if interactor.GetKeyCode() == 'n':
        print("n pressed")
        observeBoxWidget(img, box_widget, event)

def getBoundingBox(img):
    bounds = img.GetExtent()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    bbox = [bounds[0]*spacing[0] + origin[0], bounds[1]*spacing[0] + origin[0], 
            bounds[2]*spacing[1] + origin[1], bounds[3]*spacing[1] + origin[1], 
            bounds[4]*spacing[2] + origin[2], bounds[5]*spacing[2] + origin[2]]
    return bbox

def observeBoxWidget(img, box_widget, event):
    print("BoxWidget Event", event)
    
    reslice = vtk.vtkImageReslice()
    reslice.SetInterpolationModeToCubic()
    # Do not use the transform from the box widget
    # https://discourse.vtk.org/t/vtkimagereslice-with-vtkboxwidget/14776
    # trans = vtk.vtkTransform()
    # box_widget.GetTransform(trans)
    # reslice.SetInterpolationModeToLinear()
    # reslice.SetResliceTransform(trans)
    # reslice.TransformInputSamplingOn()
    
    reslice.SetInputData(img)
    
    # To find the new volume extent we need to calculate the distance between
    # the 3 parallel plane pair distance of the box
    
    planes = vtk.vtkPlanes()
    box_widget.GetPlanes(planes)
    vmath = vtk.vtkMath()
    vec = vtk.vtkVector3d()           
    # planes 0 and 1 are parallel and originally placed with normal parallel to x axis
    # planes 2 and 3 are parallel and originally placed with normal parallel to y axis
    # planes 4 and 5 are parallel and originally placed with normal parallel to z axis
    tshape = [0, 0, 0]
    origs = [0,0,0]
    for i in range(planes.GetNumberOfPlanes()):
        # get the location of the first plane on the various axes: 0, 2, 4
        if i in [0,2,4]:
            origs[i//2] = int(planes.GetPlane(i).GetOrigin()[i//2])
        for j in range(planes.GetNumberOfPlanes()):
            if i != j:
                # find if planes are parallel by checking if the normals are parallel
                vmath.Cross(planes.GetPlane(i).GetNormal(), planes.GetPlane(j).GetNormal(), vec)
                # print (f"direction {direction}, vec {vec}")
                # print ("are the norms parallel? {}".format(np.linalg.norm(direction)))
                if vmath.Norm(vec) < 1e-5:
                    # here we should be only with (0, 1), (2, 3), (4, 5) pairs
                    # print ("Direction between plane {} and plane {} is {}".format(i, j, direction))
                    orig = planes.GetPlane(j).GetOrigin()
                    # print ("Plane {} origin {}".format(j, orig))
                    # print ("Plane {} origin {}".format(i, planes.GetPlane(i).GetOrigin()))
                    dist = planes.GetPlane(i).DistanceToPlane(orig)
                    print ("Distance between plane {} and plane {} is {} {}".format(i, j, dist, dist * img.GetSpacing()[i//2]))
                    # this loop gets both (0, 1) and (1, 0) pairs
                    tshape[i//2] = dist * img.GetSpacing()[i//2]
                
    print (f"origs {origs}")
    extent = [int(el) + int(origs[i//2]) for i,el in enumerate([0, tshape[0] , 0, tshape[1] , 0, tshape[2]])]
    print ("Target shape {}, total number of voxels {}".format(tshape, tshape[0]*tshape[1]*tshape[2]))
    print ("Target extent {}".format(extent))

    
    reslice.SetResliceAxesOrigin(*origs)
    plane_dir_cos = [planes.GetPlane(1).GetNormal(), planes.GetPlane(3).GetNormal(), planes.GetPlane(5).GetNormal()]
    reslice.SetResliceAxesDirectionCosines(*plane_dir_cos)

    reslice.SetOutputExtent(*extent)
    reslice.SetOutputOrigin(0,0,0)
    orig_spacing = img.GetSpacing()
    print ("Original spacing", orig_spacing)
    reslice.SetOutputSpacing(*orig_spacing)
    # reslice.SetOutputSpacing(*[ j / i for i,j in zip(img.GetDimensions(), tshape)])
    reslice.AutoCropOutputOff()
    
    reslice.Update()
    
    # print (f"ResliceAxesOrigin {reslice.GetResliceAxesOrigin()} {origs}")
    # print (f"ResliceAxesDirectionCosines {reslice.GetResliceAxesDirectionCosines()} {plane_dir_cos}")
    # print (f"reslice extent {reslice.GetOutput().GetExtent()} {extent}")

    # cropping 


    print ("reslice extent", reslice.GetOutput().GetExtent())
    print ("reslice spacing", reslice.GetOutput().GetSpacing())
    print ("reslice origin", reslice.GetOutput().GetOrigin())

    writer = vtk.vtkMetaImageWriter()
    writer.SetInputData(reslice.GetOutput())
    writer.SetFileName('resliced.mhd')
    writer.SetCompression(False)
    writer.Write()




img = create_sample_image([64,64,64])
add_sphere(img, 9, [10, 10, 10])
add_cylinder(img, 5, [10, 50], [20, 40])
add_parallelogram(img, [30, 50], [40, 50], [10, 50])

style = vtk.vtkInteractorStyleTrackballCamera()
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.SetSize(800, 800)
renWin.SetPosition(200,100)
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.SetInteractorStyle(style)

opacityTransferFunction = vtk.vtkPiecewiseFunction()
opacityTransferFunction.AddPoint(20, 0.0);
opacityTransferFunction.AddPoint(255, 0.2);

# Create transfer mapping scalar value to color
colorTransferFunction = vtk.vtkColorTransferFunction()
colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0);
colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0);
colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0);
colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0);
colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.2, 0.0);

  # The property describes how the data will look
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorTransferFunction)
volumeProperty.SetScalarOpacity(opacityTransferFunction)
volumeProperty.ShadeOn()
volumeProperty.SetInterpolationTypeToLinear()

# The mapper / ray cast function know how to render the data
volumeMapper = vtk.vtkSmartVolumeMapper()
volumeMapper.SetInputData(img)

# The volume holds the mapper and the property and
# can be used to position/orient the volume
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

ren.AddVolume(volume)
ren.SetBackground(1,1,1);
# ren.GetActiveCamera().Azimuth(45);
# ren.GetActiveCamera().Elevation(30);
ren.ResetCameraClippingRange();
ren.ResetCamera();

# axis orientation widget
om = vtk.vtkAxesActor()
ori = vtk.vtkOrientationMarkerWidget()
ori.SetOutlineColor(0.9300, 0.5700, 0.1300)
ori.SetInteractor(iren)
ori.SetOrientationMarker(om)
ori.SetViewport(0.0, 0.0, 0.4, 0.4)
ori.SetEnabled(1)
ori.InteractiveOff()

# Add box widget to show the original volume extent
bw = vtk.vtkBoxWidget()
bw.SetInteractor(iren)
bw.HandlesOn()
bw.TranslationEnabledOn()
bw.RotationEnabledOn()
bw.GetOutlineProperty().SetColor(0.3,0.3,0.7)
bw.OutlineCursorWiresOff()
bw.SetPlaceFactor(1)
bw.KeyPressActivationOff()

bbox = getBoundingBox(img)
bw.PlaceWidget(*bbox)
bw.On()

# add observer to resample and save the resampled volume
obsf = functools.partial(observeCharEvent, img, bw, iren)
# bw.AddObserver("EndInteractionEvent", obsf)
style.AddObserver("CharEvent", obsf, 2)


iren.Start()
