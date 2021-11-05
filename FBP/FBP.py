#%%
from scipy.ndimage import rotate as sprotate
from cil.framework import ImageGeometry
from cil.framework import AcquisitionGeometry
from cil.framework import ImageData
from scipy.fft import fft, ifft, fftfreq
import numpy as np
from cil.utilities.display import show2D
# from cil.utilities import dataexample
from PIL import Image
    

def spread_detector_line(i, data, out):
    line = data.as_array()[i]
    # if out[:,0,:].shape != line.shape:
    #     width = 
    #     line = np.pad(line, width, mode='constant', constant_values=0)
    for i in range(out.shape[1]):
        out[i] = line




#%%

def rotate(ndarray, angle, centre=None, reshape=False, backend='scipy'):
    if backend == 'scipy':
        return sprotate(ndarray, angle, reshape=reshape)
    elif backend == 'pillow':
        im = Image.fromarray(ndarray)
        if centre is not None:
            pivot = [img//2 + el for img, el in zip(ndarray.shape, centre)]
        else:
            pivot = centre
        rot = np.asarray(
            im.rotate(angle, center=pivot, translate=None, fillcolor=0)
        )
        return rot


def BP(data, ig):
    '''Backward projection for 2D parallel beam'''
    spread = ig.allocate(0)
    spreadarr = spread.as_array()
    recon = ig.allocate(0)
    reconarr = recon.as_array()
    cor = data.geometry.config.system.rotation_axis.position
    try:
        np.testing.assert_array_equal(cor, np.zeros((3,), dtype=np.float32))
        centre = cor
        backend='pillow'
    except AssertionError:
        centre = None
        backend='pillow'
    print (centre, cor, backend)
    for i,angle in enumerate(data.geometry.angles):
        spread_detector_line(i, data, spreadarr)
        reconarr += rotate(spreadarr, angle, centre=centre, reshape=False, backend=backend)
    recon.fill(reconarr)
    return recon

def FP(image, ag):
    '''Forward projection for 2D parallel beam
    
    only works for centred data'''
    acq = ag.allocate(0)
    cor = acq.geometry.config.system.rotation_axis.position
    centre = None
    backend='scipy'
    for i,angle in enumerate(acq.geometry.angles):
        # TODO: check why it is -angle
        fp = rotate(image.as_array(), -angle, centre=centre, reshape=False, backend=backend)
        # TODO: axis, why 0?
        acq.fill(np.sum(fp, axis=0), angle=i)
    return acq


# %%

def rotate_image(spreadarr, angle , centre):
    im = Image.fromarray(spreadarr)

    pivot = [img//2 + el for img, el in zip(spreadarr.shape, centre)]
    rotated = im.rotate(angle, center=pivot, fillcolor=0)
    return rotated


def FBP(data, ig, backend='scipy'):
    '''Filter-Back-Projection
    
    for a nice description https://www.coursera.org/lecture/cinemaxe/filtered-back-projection-part-2-gJSJh
    '''
    spread = ig.allocate(0)
    spreadarr = spread.as_array()
    recon = ig.allocate(0)
    reconarr = recon.as_array()
    
    cor = data.geometry.config.system.rotation_axis.position
    print (cor)
    try:
        np.testing.assert_array_equal(cor, np.zeros((3,), dtype=np.float32))
        centre = cor
    except AssertionError:
        centre = None
    

    l = spread.get_dimension_size('horizontal_x')
    # create |omega| filter
    freq = fftfreq(l)
    fltr = np.asarray( [ np.abs(el) for el in freq ] )
    
    for i,angle in enumerate(data.geometry.angles):
        line = data.get_slice(angle=i).as_array()
        # apply filter in Fourier domain
        lf  = fft(line)
        lf3 = lf * fltr
        bck = ifft(lf3)
        # backproject
        for j in range(recon.shape[1]):
            spreadarr[j] = bck.real
        # should use https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate
        
        reconarr += rotate(spreadarr, angle, centre=centre, reshape=False, backend=backend)
        
    recon.fill(reconarr)
    return recon
#%%


N = 255
ig = ImageGeometry(N,N)

ag = AcquisitionGeometry.create_Parallel2D()
ag.set_panel(N)

angles = np.linspace(0, 360, 100)
ag.set_angles(angles, angle_unit='degree')


#%% Create phantom

ig = ImageGeometry(N,N)
# TomoPhantom.get_ImageData??
# phantom = TomoPhantom.get_ImageData(12, ig)
kernel_size = voxel_num_xy = N
kernel_radius = (kernel_size ) // 2
y, x = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]

circle1 = [5,0,0] #r,x,y
dist1 = ((x - circle1[1])**2 + (y - circle1[2])**2)**0.5

circle2 = [5,80,0] #r,x,y
dist2 = ((x - circle2[1])**2 + (y - circle2[2])**2)**0.5

circle3 = [25,0,80] #r,x,y
dist3 = ((x - circle3[1])**2 + (y - circle3[2])**2)**0.5

mask1 =(dist1 - circle1[0]).clip(0,1) 
mask2 =(dist2 - circle2[0]).clip(0,1) 
mask3 =(dist3 - circle3[0]).clip(0,1) 
phantomarr = 1 - np.logical_and(np.logical_and(mask1, mask2),mask3)
print (phantomarr.shape)

phantom = ImageData(phantomarr, deep_copy=False, geometry=ig, suppress_warning=True)
show2D(phantom)

# Do something with it
fwd = FP(phantom, ag)
data = fwd
# recon = FBP(data, data.geometry.get_ImageGeometry(), backend='pillow')
recon = BP(data, data.geometry.get_ImageGeometry())
show2D([data, recon])

recon2 = FBP(data, phantom.geometry)
show2D([phantom, recon, recon2], title=['phantom', 'Back Projection', 'fbp'])

# #%%



# dls = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
# print ("##################################### DLS", dls.geometry)
# data = dls.log()
# data *= -1
# data = data.get_slice(vertical='centre')
# ig = data.geometry.get_ImageGeometry()
# recon = BP(data, ig)
# show2D(recon, title=['Back Projection'])

# show2D([phantom, recon, recon2, \
#         data, fwd, fwd - data], \
#     title=['phantom', 'Back-Projection', 'FBP', \
#            'ASTRA FWD', 'FP FWD', 'DIFF (FP - ASTRA)_FWD'],\
#         cmap='gist_earth', num_cols=3)
# from cil.utilities.display import show_geometry


# dls = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
# print ("##################################### DLS", dls.geometry)
# data = dls.log()
# data *= -1
# show_geometry(data.geometry)
# # dls2d = dls.get_slice(vertical=70)
# # show2D(dls.get_slice(vertical='centre'))
# #%%
# from cil.processors import Slicer, CentreOfRotationCorrector
# data = CentreOfRotationCorrector.xcorr(slice_index='centre', projection_index=0, ang_tol=0.1)(data)
# print ("##################################### Centred", data.geometry)
# # dls2d = data.get_slice(vertical=70)
# # print ("##################################### Centred", dls2d.geometry)
# # get centre of rotation
# cor = data.geometry.config.system.rotation_axis.position
# px = data.geometry.config.panel.pixel_size[0]
# cor = [cor[0] * px , cor[1] * px]
# if cor[0] <= 0:
#     pad = (0, 2* int(cor[0]))
# else:
#     pad = (2* int(cor[0]), -1)
# print ("CENTRE of ROT", cor, pad)

# from cil.processors import Slicer
# dls2d = Slicer(roi={'horizontal': pad })(dls.get_slice(vertical=70))
# data = dls2d.log()
# data *= -1
# # dls2d.geometry = dls.get_slice(vertical=70).geometry.copy()

# # # dls2d.geometry.config.system.rotation_axis.position = data.geometry.config.system.rotation_axis.position[:-1]
# # dls_ig = dls2d.geometry.get_ImageGeometry()

# # print ("rotation axis", data.geometry.config.system.rotation_axis.position)

# # %%
