from scipy.ndimage import rotate
from cil.framework import ImageGeometry
from cil.framework import AcquisitionGeometry
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, fftfreq, ifftshift
import numpy as np
from cil.utilities.display import show2D
from cil.utilities import dataexample
from PIL import Image
    

def spread_detector_line(i, data, out):
    line = data.as_array()[i]
    for i in range(out.shape[1]):
        out[i] = line

def spread_and_filter_detector_line(i, data, filt, out):
    line = data.as_array()[i]
    for i in range(out.shape[1]):
        out[i] = line * filt

def ramp_filter(size):
    ramp = np.zeros((size))
    middle = size // 2
    
    if size % 2 != 0:
        middle = size // 2 + 1
     
    half_ramp = np.linspace(0,1,middle, True)
    ramp[middle:] = half_ramp
    ramp[:middle] = half_ramp[::-1]
    return ramp

def fourier_filter_data(data):
    from scipy.fft import fft, ifft
    out = data.copy()
    
    for i,angle in enumerate(data.geometry.angles):
        line = data.as_array()[i]
        
        ramp = np.asarray(ramp_filter(line.size), dtype=np.complex)
        ramp = fftshift(ramp)
#         ramp.imag = ramp.real

        lf = fftshift(fft(line))
#         plt.plot(lf.real)
        lf *= ramp
        lf = ifft(lf)
        out.fill(lf, angle=i)
    return out


N = 255
ig = ImageGeometry(N,N)

ag = AcquisitionGeometry.create_Parallel2D()
ag.set_panel(N)

angles = np.linspace(0, 360, 100)
ag.set_angles(angles, angle_unit='degree')


#%% Create phantom
# from cil.plugins import TomoPhantom
from cil.framework import ImageGeometry, ImageData
N = 255
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
# show2D(phantom)

#%%
from cil.io import NEXUSDataReader

reader = NEXUSDataReader()
reader.set_up(file_name='phantom.nxs')
data = reader.read()


def BP(data, ig):
    '''Backward projection for 2D parallel beam'''
    spread = ig.allocate(0)
    spreadarr = spread.as_array()
    recon = ig.allocate(0)
    reconarr = recon.as_array()
    
    for i,angle in enumerate(data.geometry.angles):
        spread_detector_line(i, data, spreadarr)
        reconarr += rotate(spreadarr, angle , reshape=False)
    recon.fill(reconarr)
    return recon

def FP(image, ag):
    '''Forward projection for 2D parallel beam'''
    acq = ag.allocate(0)
    
    for i,angle in enumerate(data.geometry.angles):
        # TODO: check why it is -angle
        fp = rotate(image.as_array(), -angle , reshape=False)
        # TODO: axis, why 0?
        acq.fill(np.sum(fp, axis=0), angle=i)
    return acq

def spread_and_filter_detector_line(i, data, filt, out):
    line = data.as_array()[i]
    for i in range(out.shape[1]):
        out[i] = line * filt

def rotate_image(spreadarr, angle , centre):
    im = Image.fromarray(spreadarr)

    pivot = [img//2 + el for img, el in zip(spreadarr.shape, centre)]
    rotated = im.rotate(angle, center=pivot, fillcolor=0)
    return rotated


def FBP(data, ig):
    '''Filter-Back-Projection'''
    spread = ig.allocate(0)
    spreadarr = spread.as_array()
    recon = ig.allocate(0)
    reconarr = recon.as_array()
    
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
        reconarr += rotate(spreadarr, angle , reshape=False)
        
    recon.fill(reconarr)
    return recon

recon = BP(data, ig)
recon2 = FBP(data, ig)
fwd = FP(phantom, data.geometry)
show2D([phantom, recon, recon2, \
        data, fwd, fwd - data], \
    title=['phantom', 'Back-Projection', 'FBP', \
           'ASTRA FWD', 'FP FWD', 'DIFF (FP - ASTRA)_FWD'],\
        cmap='gist_earth', num_cols=3)

dls = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
dls2d = dls.get_slice(vertical=70)

from cil.processors import Slicer, CentreOfRotationCorrector
data = CentreOfRotationCorrector.xcorr(slice_index='centre')(dls)
print (data.geometry)
dls2d = data.get_slice(vertical=70)
# get centre of rotation
    # cor = data.geometry.config.system.rotation_axis.position
    # px = data.geometry.config.panel.pixel_size[0]
    # cor = [cor[0] * px , cor[1] * px]
    
from cil.processors import Slicer
dls2d = Slicer(roi={'horizontal': (14,-1)})(dls2d)

# dls2d.geometry.config.system.rotation_axis.position = data.geometry.config.system.rotation_axis.position[:-1]
dls_ig = dls2d.geometry.get_ImageGeometry()

print ("rotation axis", data.geometry.config.system.rotation_axis.position)

recon = FBP(dls2d, dls_ig)

show2D([dls2d, recon])
