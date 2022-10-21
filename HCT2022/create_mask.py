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
# define the function to be fitted
def f(x, data, rim, circle):
    '''Objective function which multiplies the circumference with the data.
    
    Since we are doing minimisation we multiply by -1 and calculate the log as
    this product is expected to change sharply'''
    create_circumference(x, circle.array, * circle.shape, rim)
    circle *= data
    return - np.log(circle.sum())

# %%
# create initial guess by making a circumference centred in the middle of the 
# image space and with max radius
x0 = np.asarray([ig.shape[0]/2, ig.shape[0]/2 , ig.shape[1]/2])
# having a large rim helps finding the minimum
rim = 30

circ = ig.allocate(0)
create_circumference(x0, circ.array, *circ.shape, rim)
import matplotlib.pyplot as plt
plt.imshow(mag.array, cmap='inferno', vmax = 200,  origin='lower')
plt.imshow(circ.array , cmap='gray', alpha=0.4,origin='lower')

plt.show()
#%%
# run minimisation
res = minimize(f, x0, args=(mag, rim, circ), method='Nelder-Mead', tol=1e-6)
print(res.x)
circ = ig.allocate(0)
xx = res.x * 1
# The fit returns the internal edge of the circumference
xx[0] = xx[0] - rim
# xx is our solution

#%%
# create the circumference for plotting
create_circumference(xx, circ.array, *circ.shape, 3)

# overlay the circumference to the gradient magnitude
# default origin in show2D is lower
plt.imshow(mag.array, cmap='inferno', vmax=200, origin='lower')
plt.imshow(circ.array , cmap='gray', alpha=0.4, origin='lower')
plt.show()
# %%
# overlay the circumference to the full data reconstruction for evaluation
fdk = FDK(full_data, ig)
recon = fdk.run()
#%%
# default origin in show2D is lower
plt.imshow(recon.array, cmap='gray', origin='lower')
plt.imshow(circ.array , cmap='inferno', alpha=0.5, origin='lower')
plt.show()
# %%
