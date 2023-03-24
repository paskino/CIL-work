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
import matplotlib.pyplot as plt
import os
from cil.processors import Slicer



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
def fill_circular_mask(rc, array, value, N, M):

    for i in numba.prange(M):
        for j in numba.prange(N):
            d = np.sqrt( (i-rc[1])*(i-rc[1]) + (j-rc[2])*(j-rc[2]))
            if d<rc[0]:
                array[i,j] = value
            else:
                array[i,j] = 0
# find each point x,y in the mask
@numba.jit(nopython=True)
def get_coordinates_in_mask(mask, N, M, out, value=1):
    '''gets the coordinates of the points in a mask'''
    k = 0
    for i in numba.prange(M):
        for j in numba.prange(N):
            if mask[i,j] == value:
                out[0][k] = i
                out[1][k] = j
                k += 1

def calculate_gradient_magnitude(data):
    '''calculates the magnitude of the gradient of the input data'''
    grad = GradientOperator(data.geometry)
    mag = grad.direct(data)
    mag = mag.get_item(0).power(2) + mag.get_item(1).power(2)
    return mag

@numba.jit(nopython=True)
def set_mask_to_zero(mask, where, where_value, N, M):
    for i in numba.prange(M):
        for j in numba.prange(N):
            if where[i,j] == where_value:
                mask[i,j] = 0

def find_circle_parameters(data, ig):
    '''Creates a circular mask for the reconstruction in the specified ImageGeometry
    
    Parameters:
    -----------

    data: input data, sinogram
    ig: reconstruction volume geometry
    value: value to set the mask to, defaults to 1.
    return_all: boolean to ask to return both the circular mask and the edges
                of the circle

    Returns:
    --------
    ImageData with a circular mask set at value. Optionally returns the edges of
    the circle if return_all is set to True
    '''

    # thresh = threshold_otsu(data.array)

    # bmask = data.array > thresh

    # data_mask = data.geometry.allocate(0)
    # data_mask.array[bmask] = 1.

    ig = data.geometry.get_ImageGeometry()
    A = ProjectionOperator(ig, data.geometry)
    recon = FDK(data, ig).run()
    # recon = A.adjoint(data)

    # find the good edge on the FDK reconstruction
    # 1) calculate the gradient magnitude
    # 2) thresholding it with an otsu threshold
    # 3) generate a mask with that threshold

    mag = calculate_gradient_magnitude(recon)
    # grad = GradientOperator(ig)
    # mag = grad.direct(recon)
    # mag = mag.get_item(0).power(2) + mag.get_item(1).power(2)
    
    # initial binary mask
    thresh = threshold_otsu(mag.array)
    binary_mask = mag.array > thresh

    mask = ig.allocate(0.)
    previous_num_datapoints = mask.size
    num_iterations = 20
    delta = 4 # pixels
    value = 1
    for i in range(num_iterations):
        
        maskarr = mask > 0

        set_mask_to_zero(binary_mask, maskarr, value, *binary_mask.shape)
        
        # find the coordinates of the points in the binary mask
        num_datapoints = np.sum(binary_mask)
        print ("iteration {}, num_datapoints {}, sum(mask) {}".format(i, num_datapoints, np.sum(maskarr)))
        if num_datapoints < previous_num_datapoints:
            previous_num_datapoints = num_datapoints
        else:
            return np.asarray([r, x0, y0])
        out = np.zeros((2, num_datapoints), dtype=int)

        get_coordinates_in_mask(binary_mask, *binary_mask.shape, out)
        # x,y = out

        # fit a circle to the points
        x0,y0,r = fit_circle(*out)

        # fill a mask and return it
        mask.fill(0)
        mycircle = np.asarray([r-delta, x0, y0])

        fill_circular_mask(mycircle, mask.array, value, *mask.shape)



    
    return np.asarray([r, x0, y0])
    
    

#%%

full_data = load_htc2022data(os.path.abspath('C:/Users/ofn77899/Data/HTC2022/htc2022_tc_full.mat'))
#%%


ig = full_data.geometry.get_ImageGeometry()
recon_f = FDK(full_data, ig).run()
#%%
# Reduce the dataset 
# these are indices in the angles array
# start at offset, use num_angle
offset = 0
num_angles = 60

#%%

roi = {'angle':(offset, offset+num_angles, None)}
data = Slicer(roi=roi)(full_data)


#%%

ig = data.geometry.get_ImageGeometry()
recon = FDK(data, ig).run()

#%%
circle_parameters = find_circle_parameters(data, ig)
mask = ig.allocate(0)
fill_circular_mask(circle_parameters, mask.array, 1, *mask.shape)

#%%
circle = calculate_gradient_magnitude(mask)
plt.imshow(recon_f.array, cmap='gray_r', vmax = 1,  origin='lower')
# plt.imshow(mag.array, cmap='gray_r', vmax = 1, alpha=0.2,  origin='lower')
plt.imshow(circle.array , cmap='gray_r', alpha=0.4, origin='lower')
plt.show()

# %%

# %%
