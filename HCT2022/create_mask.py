#%%
import numpy as np
from cil.utilities.display import show2D
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.recon import FDK
import scipy.io
import numba
from cil.utilities.display import show2D, show_geometry
#use Otsu threshold
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from cil.plugins.tigre import ProjectionOperator
from cil.optimisation.operators import GradientOperator
from scipy.optimize import minimize


#%%

def load_htc2022data(filename):
    mat = scipy.io.loadmat(filename)
    sinogram = mat['CtDataFull'][0][0][1].astype('float32')
    num_angles = sinogram.shape[0]
    num_dets = sinogram.shape[1]
    source_center = 410.66
    source_detector = 553.74
    det_pixel_size = 0.2
    angles = np.linspace(0,360,num_angles,endpoint=True)
    ag = AcquisitionGeometry.create_Cone2D(source_position=[0,-source_center], 
                                           detector_position=[0,source_detector-source_center])\
        .set_panel(num_pixels=num_dets, pixel_size=0.2)\
        .set_angles(angles=-angles, angle_unit='degree')
    data = AcquisitionData(sinogram, geometry=ag)
    return data

def fit_circle(x,y):
    '''Circle fitting by linear and nonlinear least squares in 2D
    
    Parameters
    ----------
    x : array with the x coordinates of the data
    y : array with the y coordinates of the data. It has to have the
        same length of x.

    Returns
    -------
    x0 : x coordinate of the centre
    y0 : y coordinate of the centre
    r : radius of the circle
    
    References
    ----------

    Journal of Optimisation Theory and Applications
    https://link.springer.com/article/10.1007/BF00939613
    From https://core.ac.uk/download/pdf/35472611.pdf
    '''
    if len(x) != len(y):
        raise ValueError('X and Y array are of different length')
    data = np.vstack((x,y))

    B = np.vstack((data, np.ones(len(x))))
    d = np.sum(np.multiply(data,data), axis=0)

    res = np.linalg.lstsq(B.T,d, rcond=None)
    y = res[0]
    x0 = y[0] * 0.5 
    y0 = y[1] * 0.5
    r = np.sqrt(x0**2 + y0**2 + y[2])

    return (x0,y0,r)

@numba.jit(nopython=True)
def create_circle(rc, array, value, N, M):

    for i in numba.prange(M):
        for j in numba.prange(N):
            d = np.sqrt( (i-rc[1])*(i-rc[1]) + (j-rc[2])*(j-rc[2]))
            if d<rc[0]:
                array[i,j] = value
            else:
                array[i,j] = 0

#%%
import os

full_data = load_htc2022data(os.path.abspath('C:/Users/ofn77899/Data/HTC2022/htc2022_tb_full.mat'))

# Reduce the dataset 
# these are indices in the angles array
# start at offset, use num_angle
offset = 90
num_angles = 45

#%%
from cil.processors import Slicer

roi = {'angle':(offset, offset+num_angles, None)}
data = Slicer(roi=roi)(full_data)


#%%

thresh = threshold_otsu(data.array)

bmask = data.array > thresh

data_mask = data.geometry.allocate(0)
data_mask.array[bmask] = 1.

#%%

show2D([data, data_mask])

#%%
ig = data.geometry.get_ImageGeometry()
A = ProjectionOperator(ig, data.geometry)

# backproject the mask of the data
backmask = A.adjoint(data_mask)
show2D(backmask)

#%%
# find the good edge on the backprojected mask by
# 1) calculate the gradient magnitude
# 2) thresholding it with an otsu threshold
# 3) generate a mask with that threshold


grad = GradientOperator(ig)
mag = grad.direct(backmask)
mag = mag.get_item(0).power(2) + mag.get_item(1).power(2)
#%%
thresh = threshold_otsu(mag.array)
binary_mask = mag.array > thresh
mask = mag.geometry.allocate(0)
mask.array[binary_mask] = 1.

show2D([mag, mask])

#%%
# find each point x,y in the mask
@numba.jit(nopython=True)
def get_points_in_mask(mask, N, M, out, value=1):
    '''gets the coordinates of the points in a mask'''
    k = 0
    for i in numba.prange(M):
        for j in numba.prange(N):
            if mask[i,j] == value:
                out[0][k] = i
                out[1][k] = j
                k += 1

num_datapoints = np.sum(binary_mask)
out = np.zeros((2, num_datapoints), dtype=int)

get_points_in_mask(binary_mask, *binary_mask.shape, out)
x,y = out



x0,y0,r = fit_circle(x,y)

#%%
@numba.jit(nopython=True)
def create_circumference(rc, circle, N, M, rim):
    '''create a circumference 
    
    rc = [radius, x0, y0]
    circle: data to fill
    N,M: circle shape
    rim: size in pixels of the rim of the circumference
    '''
    ret = 0
    for i in numba.prange(M):
        for j in numba.prange(N):
            d = np.sqrt( (i-rc[1])*(i-rc[1]) + (j-rc[2])*(j-rc[2]))
            # if np.isclose(d, rc[0], atol=1e-5):
            # if np.abs(d - rc[0]) <= (atol + rtol * np.abs(rc[0])):
            if d < rc[0] and d > rc[0] - rim:
                circle[i,j] = 1
            else:
                circle[i,j] = 0



# %%
mycircle = np.asarray([r,x0,y0])

rim = 1

circ = ig.allocate(0)
create_circumference(mycircle, circ.array, *circ.shape, rim)
import matplotlib.pyplot as plt
plt.imshow(mag.array, cmap='inferno', vmax = 200,  origin='lower')
plt.imshow(circ.array , cmap='gray', alpha=0.4,origin='lower')

plt.show()
#%%
new_mask = ig.allocate(0)

create_circle(mycircle, new_mask.array, 1, *new_mask.shape)
plt.imshow(mag.array, cmap='inferno', vmax = 200,  origin='lower')
plt.imshow(new_mask.array , cmap='gray', alpha=0.4,origin='lower')

plt.show()

# %%
