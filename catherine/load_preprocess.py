#%%
from cil.io import ZEISSDataReader
from cil.processors import Slicer
from cil.recon import FDK
from cil.utilities.display import DataContainer, show2D
import os
import olefile
import dxchange
import numpy as np
from cil.framework import AcquisitionData
from tqdm import tqdm
from cil.io import NEXUSDataWriter
from cil.processors import TransmissionAbsorptionConverter, Normaliser, CentreOfRotationCorrector
from cil.framework import DataContainer
from cil.plugins.astra import FBP

#%%

fname = os.path.abspath("/home/ofn77899/Data/kidney.txrm")

#%
#############
#Correction for multiple reference files
# %%


try:
    ole = olefile.OleFileIO(fname)
except IOError:
    print('No such file or directory: %s', fname)

# %%
ole_metadata = ole.listdir()
# %%
for i in ole_metadata:
    print(i)
# %%
from dxchange.reader import _read_ole_value, _read_ole_image

# a = _read_ole_value(ole, 'ImageInfo/referencefile', '<260s')
# print(a)
# %%

def concat_to_string(lst):
    ret = ''.join([s + "/"for s in lst])
    return ret[:-1]


_multi_ref_image = [ ['MultiReferenceData', 'Image1'],
                    ['MultiReferenceData', 'Image2'],
                    ['MultiReferenceData', 'Image3'],
                    ['MultiReferenceData', 'Image4']]

_multi_ref_image_data_type = ['MultiReferenceData', 'ImageInfo', 'DataType']
# %%
multi_ref_image_data_type = _read_ole_value(ole, concat_to_string(_multi_ref_image_data_type), '<1I')

#%%
reader = ZEISSDataReader(fname)
metadata = reader.get_metadata()

# print(metadata)

# %%
ref_images = []
for el in _multi_ref_image:
    ref_images.append(_read_ole_image(ole, concat_to_string(el), metadata,
                                      multi_ref_image_data_type))

# %%
# show2D(ref_images, cmap="viridis", fix_range=False)
# # %%
# show2D([el / ref_images[0] for el in ref_images[1:]], cmap="viridis")
# # %%
# average_ref_image = sum(ref_images) / len(ref_images)
# show2D(average_ref_image)
# %%
# reader._metadata['reference'] = average_ref_image
# #%%
# data = reader.read()

# #%%

# proj = data.get_slice(angle=800)
# proj_n = proj / average_ref_image
# show2D([proj, average_ref_image, proj_n], title=["Projection", "Reference Image", "Normalised Projection"])



# %%

dataarr, _ = dxchange.read_txrm(fname,None)
#%%

N = 800
ref_num = 3
# norm_dataarr = dataarr / average_ref_image
show2D([dataarr[N,:,:], ref_images[ref_num], dataarr[N,:,:] / ref_images[ref_num]], title=["Projection", f"Reference Image {ref_num}", "Normalised Projection"])
#%%

# data = data / self._metadata['reference']
norm_data = np.empty(dataarr.shape, dtype=np.float32)

for num in tqdm(range(reader._metadata['number_of_images'])):
    norm_data[num,:,:] = np.asarray(dataarr[num,:,:], dtype=np.float32) / ref_images[ref_num]
    norm_data[num,:,:] = np.roll(norm_data[num,:,:], \
        (int(reader._metadata['x-shifts'][num]),int(reader._metadata['y-shifts'][num])), \
        axis=(1,0))

acq_data = AcquisitionData(array=norm_data, deep_copy=False, geometry=reader._geometry.copy())
#%%
writer = NEXUSDataWriter(acq_data, "kidney.nxs")
writer.write()
#%%

data2d =acq_data.get_slice(vertical="centre")

writer = NEXUSDataWriter(data2d, "kidney_2D.nxs")
writer.write()
#%%

show2D(data2d)
# %%

converter = TransmissionAbsorptionConverter()
converter.set_input(data2d)
data2d_abs = converter.get_output()
show2D(data2d_abs)

#%%%
find_cor = False
if find_cor:
    # Find the centre of rotation manually
    
    array_list = []
    pixel_offsets = [3, 4, 5, 6, 7]
    ig = data2d_abs.geometry.get_ImageGeometry()
    fbp = FBP(ig, data2d_abs.geometry, device="gpu")

    for p in tqdm(pixel_offsets):
        data2d_abs.geometry.set_centre_of_rotation(p, distance_units='pixels')
        fbp = FBP(ig, data2d_abs.geometry, device="gpu")
        # data_slice = data_test.get_slice(vertical=vertical_slice)
        fbp.set_input(data2d_abs)
        array_list.append(fbp.get_output().as_array())

    DC = DataContainer(np.stack(array_list, axis=0), dimension_labels=tuple(['Centre of rotation offset']) + ig.dimension_labels)

#%%

if find_cor:
    # from cil.utilities.jupyter import islicer
    # islicer(DC, title=tuple(['Centre of rotation offset: ' + str(p)  + ', index: ' for p in pixel_offsets]))

    show2D([DC.array[el,445:726,676:866] for el in range(DC.array.shape[0])],
            title=['Centre of rotation offset: ' + str(p)  + ', index: ' for p in pixel_offsets], 
            fix_range=(-0.05, 0.5))



# %%
# manually found centre of rotation to be 4 pixels
cor = 4
data2d_abs.geometry.set_centre_of_rotation(4, distance_units='pixels')

ig = data2d_abs.geometry.get_ImageGeometry()
fbp = FBP(ig, data2d_abs.geometry, device="gpu")
# %%
fbp.set_input(data2d_abs)
recon_fbp = fbp.get_output()

# %%
recon_fbp.apply_circular_mask(radius=0.97, in_place=True)

# %%
show2D(recon_fbp.array[300:700,700:1200], fix_range=False)

# %%
show2D(recon_fbp, fix_range=(-0.05, 0.3))
# %%

def __read_ole_value(ole, label, struct_fmt):
    value = None
    if ole.exists(label):
        stream = ole.openstream(label)
        data = stream.read()
        # value = struct.unpack(struct_fmt, data)
        return data

energy = ['AMC', 'ImageInfo', 'Energy']
energy_val = __read_ole_value(ole, concat_to_string(energy), '<i')
print(f"Energy: {energy_val} eV")
# %%
import numpy as np

b = energy_val
arr_i32 = np.frombuffer(b, dtype='<i4')
arr_f32 = np.frombuffer(b, dtype='<f4')
print(arr_i32.shape, arr_i32[:10], arr_i32.min(), arr_i32.max())
print(arr_f32.shape, arr_f32[:10], arr_f32.min(), arr_f32.max())
# %%
