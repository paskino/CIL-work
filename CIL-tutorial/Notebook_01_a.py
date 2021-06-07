#%%
import numpy as np
import os

from cil.io import TXRMDataReader
# from cil.plugins.tigre import FBP as tFBP
from cil.processors import TransmissionAbsorptionConverter
from cil.plugins.astra import FBP as aFBP
from cil.utilities.display import show2D
from cil.utilities.jupyter import islicer
# from cil.processors import CentreOfRotationCorrector

#%%

#%%

path = os.path.abspath('/home/edo/scratch/Dataset/')
filename = os.path.join(path, 'CCPi', 'valnut_tomo-A.txrm')

# reader = TXRMDataReader()
# reader.set_up(file_name=filename)
# data = reader.read()

# reader = TXRMDataReader(file_name=filename)
# data = reader.read()

data = TXRMDataReader(file_name=filename).read()

print("read in data type", data.dtype)
#%%
show2D(data, slice_list=('angle',800), cmap='inferno', origin='upper-right')
#%%
# get central slice
# data2d = data.get_slice(vertical='centre') 
# data2d.reorder(order='tigre')


#%% 
# Centre of rotation finder

#%%
# neg log

data = TransmissionAbsorptionConverter()(data)

#%%
from cil.utilities.jupyter import islicer
islicer(data, direction='angle', cmap='inferno', origin='upper-right')
#%%

ig = data.geometry.get_ImageGeometry()

#%%
data.reorder(order='astra')
#%%
fbp =  aFBP(ig,data.geometry)
# fbp.set_input(data2d)

# recon = fbp.get_output()
recon = fbp(data)

#%%
from cil.utilities.jupyter import islicer
islicer(recon, direction='vertical', cmap='inferno', origin='lower')
#%%
show2D([data, recon], title=['sinogram', 'ASTRA FBP'],\
    slice_list=[['vertical','centre'],['vertical','410']])

#%%
# save output to TIFF and NeXuS
from cil.io import TIFFWriter
path = os.path.abspath('/home/edo/scratch/Dataset/')
out_file = os.path.join(path, 'edo', 'recon')
writer = TIFFWriter(data=recon, file_name=out_file)
writer.write()

#%%
from cil.io import TIFFStackReader
path = os.path.abspath('/home/edo/scratch/Dataset/')
fname = os.path.join(path, 'edo')
treader = TIFFStackReader()
treader.set_up(file_name=fname)
# treader.read()

#%%
recon = treader.read_as_ImageData(ig)
#%%
from cil.utilities.jupyter import islicer
islicer(recon, direction='vertical', cmap='inferno', origin='upper-right')

#%%
from cil.io import NikonDataReader
reader = NikonDataReader()
path = os.path.abspath('/home/edo/scratch/Dataset/')
# /SparseBeads_ML_L3/CentreSlice/
reader.set_up(file_name = 
    os.path.join(path, 'SophiaBeads_64_averaged', 'SophiaBeads_64_averaged.xtekct' )
       )

xtek_sino = reader.read()

#%%
#xtek_sino = xtek_sino.get_slice(vertical='centre')


# %%
xtek_sino = TransmissionAbsorptionConverter()(xtek_sino)

#%%

ig_xtek = xtek_sino.geometry.get_ImageGeometry()

#%%
xtek_sino.reorder(order='astra')

#%%
from cil.processors import CentreOfRotationCorrector
cor = CentreOfRotationCorrector.sobel(FBP=aFBP)
cor.set_input(xtek_sino)
centred = cor.get_output()
ig_xtek = centred.geometry.get_ImageGeometry()
#%%
fbp =  aFBP(ig_xtek,centred.geometry)
# fbp.set_input(data2d)

# recon = fbp.get_output()
recon = fbp(xtek_sino)

#%%
from cil.utilities.jupyter import islicer
islicer(recon, direction='vertical', cmap='inferno', origin='lower')
# %%
