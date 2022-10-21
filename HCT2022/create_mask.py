#%%
import numpy as np
from cil.utilities.display import show2D
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.recon import FDK
import scipy.io
import numba
from cil.framework import VectorData, VectorGeometry
from cil.optimisation.operators import Operator, LinearOperator

import matplotlib.pyplot as plt

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

@numba.jit(nopython=True)
def find_trajectory(array, rc, threshold, N, M):
    '''finds the baricentre of a 1D plot'''
    for j in numba.prange(N):
        baricx = 0.
        mass = 0 
        minx = N - 1
        maxx = 0
        # miny = M - 1
        # maxy = 0
        for i in numba.prange(M):
            # print (array[j,i], threshold)
            if array[j,i] > threshold:
                baricx += i
                mass += 1 
                if minx > i:
                    minx = i
                if maxx < i:
                    maxx = i
        if mass > 0:
            baricx /= mass
        else:
            baricx = -1
    
        rc[0, j] = baricx
        rc[1, j] = minx
        rc[2, j] = maxx
        rc[3, j] = maxx - minx
        rc[4, j] = mass
    
#%%
import os

full_data = load_htc2022data(os.path.abspath('C:/Users/ofn77899/Data/HTC2022/htc2022_tb_full.mat'))

#%%
from cil.processors import Slicer

offset = 90
cone_angle = 45

roi = {'angle':(offset,offset+cone_angle,None)}
data = Slicer(roi=roi)(full_data)


#%%
from cil.utilities.display import show2D, show_geometry

show2D(data)

#%%
#use Otsu threshold
from skimage.filters import threshold_otsu
thresh = threshold_otsu(data.array)

bmask = data.array > thresh

mask = data.geometry.allocate(0)
mask.array[bmask] = 1.

#%%

show2D([data, mask])

#%%
from cil.recon import FDK
from cil.plugins.tigre import ProjectionOperator

# fdk = FDK(mask, ig)

# backmask = fdk.run()
ig = data.geometry.get_ImageGeometry()

A = ProjectionOperator(ig, data.geometry)
backmask = A.adjoint(mask)

ones = data.geometry.allocate(1)
backones = A.adjoint(ones)
show2D([backmask, backones])

# %%
fdk = FDK(data, ig)

recon = fdk.run()
show2D([backones, recon, backmask], title=['ones', 'recon', 'mask'], num_cols=3)

#%%

from cil.optimisation.operators import GradientOperator

grad = GradientOperator(ig)
mag = grad.direct(backmask)
mag2 = (mag * mag.conjugate()).sqrt()
mag3 = mag.get_item(0).power(2) + mag.get_item(1).power(2)
#%%
thresh = threshold_otsu(mag3.array)
bmask = mag3.array > thresh
mask = mag3.geometry.allocate(0)
mask.array[bmask] = 1.

show2D([mag3, mask])



#%%
@numba.jit(nopython=True)
def multiply_with_circle(rc, data, N, M, atol=1e-8, rtol=1e-5):
    '''fill an image with a circle
    
    rc = [radius, x0, y0]
    data: mask
    '''
    ret = 0
    for i in numba.prange(M):
        for j in numba.prange(N):
            d = np.sqrt( (i-rc[1])*(i-rc[1]) + (j-rc[2])*(j-rc[2]))
            if np.abs(d - rc[0]) <= (atol + rtol * np.abs(rc[0])):
                ret += data[i,j]
    return ret
@numba.jit(nopython=True)
def create_circle(rc, circle, N, M, rim):
    '''fill an image with a circle
    
    rc = [radius, x0, y0]
    data: mask
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
#%%
def create_circle_p(rc, circle, N, M):
    '''fill an image with a circle

    as above in python
    
    rc = [radius, x0, y0]
    data: mask
    '''
    ret = 0
    for i in range(M):
        for j in range(N):
            d = np.sqrt( (i-rc[1])*(i-rc[1]) + (j-rc[2])*(j-rc[2]))
            if np.isclose(d, rc[0], atol=1e-3, rtol=1e-2):
                circle[i,j] = 1
            else:
                circle[i,j] = 0
    
    
#%%

rc = np.asarray([150, 250, 250])
circle = mask * 0
create_circle(rc, circle.array, *circle.shape, 10)
show2D(circle)
# %%
show2D([circle, mask, mag3 * circle, mag3])
# %%
def f(x, data, rim, circle):
    
    create_circle(x, circle.array, * circle.shape, rim)
    circle *= data
    return - np.log(circle.sum())

print (f(rc, mag3))
# %%

from scipy.optimize import minimize
x0 = np.asarray([mag3.shape[0]/2, mag3.shape[0]/2 , mag3.shape[1]/2])
rim = 30
# x0 = np.asarray([230.64850342, 275.29027125, 275.28706522])
circle = mag3 * 0
create_circle(x0, circle.array, *circle.shape, rim)
import matplotlib.pyplot as plt
plt.imshow(mag3.array, cmap='inferno', vmax = 200,  origin='lower')
plt.imshow(circle.array , cmap='gray', alpha=0.4,origin='lower')

plt.show()
#%%
# data can be mag3 or mask
res = minimize(f, x0, args=(mag3, rim, circle), method='Nelder-Mead', tol=1e-6)
print(res.x)
circle = mag3 * 0
xx = res.x * 1
xx[0] = xx[0] - rim
create_circle(xx, circle.array, *circle.shape, 3)
#%%
# show2D([circle, mag3 * circle], cmap=['gray', 'gray_r'])
import matplotlib.pyplot as plt
# default origin in show2D is lower
plt.imshow(mag3.array, cmap='inferno', vmax=200, origin='lower')
plt.imshow(circle.array , cmap='gray', alpha=0.4, origin='lower')
plt.show()
# %%
