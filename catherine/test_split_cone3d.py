#%%
from cil.framework import AcquisitionData, ImageGeometry
from cil.utilities import dataexample
from cil.utilities.display import show_geometry, show2D
from cil.processors import Slicer, TransmissionAbsorptionConverter
from cil.recon import FDK
from cil.utilities.jupyter import islicer





# %%

def split_cone_beam_data(data, resolution=1, half="top", roi={}):
    ig = data.geometry.get_ImageGeometry(resolution=resolution)
    if half == "top":
        ig_half = ImageGeometry(voxel_num_x=ig.voxel_num_x, 
                                voxel_num_y=ig.voxel_num_y, 
                                voxel_num_z=ig.voxel_num_z//2,
                                voxel_size_x=ig.voxel_size_x, 
                                voxel_size_y=ig.voxel_size_y,
                                voxel_size_z=ig.voxel_size_z,
                                center_x=ig.center_x,
                                center_y=ig.center_y,
                                center_z=ig.voxel_num_z / 4 * ig.voxel_size_z)
        if 'bottom' in data.geometry.config.panel.origin.split("-"):
            half_range = (data.get_dimension_size("vertical")//2, data.get_dimension_size("vertical"),1)
        else:
            half_range = (0, data.get_dimension_size("vertical")//2,1)
    elif half == "bottom":
        ig_half = ImageGeometry(voxel_num_x=ig.voxel_num_x, 
                                voxel_num_y=ig.voxel_num_y, 
                                voxel_num_z=ig.voxel_num_z//2,
                                voxel_size_x=ig.voxel_size_x, 
                                voxel_size_y=ig.voxel_size_y,
                                voxel_size_z=ig.voxel_size_z,
                                center_x=ig.center_x,
                                center_y=ig.center_y,
                                center_z=-ig.voxel_num_z / 4 * ig.voxel_size_z)
        if 'bottom' in data.geometry.config.panel.origin.split("-"):
            half_range = (0, data.get_dimension_size("vertical")//2,1)
        else:
            half_range = (data.get_dimension_size("vertical")//2, data.get_dimension_size("vertical"),1)
    else:
        raise ValueError("half must be either 'top' or 'bottom'")
    # remove vertical roi if it exists and replace with half range
    roi.pop("vertical", None)
    roi['vertical'] = half_range

    slicer = Slicer(roi=roi)
    slicer.set_input(data)
    data_reduced = slicer.get_output()

    return data_reduced, ig_half

#%%
data = dataexample.SIMULATED_CONE_BEAM_DATA.get()
ta = TransmissionAbsorptionConverter()
ta.set_input(data)
data_full = ta.get_output()
#%%

data_reduced, ig_half = split_cone_beam_data(data_full, resolution=1, 
                                             half="bottom")
#%%
show_geometry(data_reduced.geometry, 
              ig_half, 
              elevation=0, 
              azimuthal=0, 
              view_distance=1)
# %%
fdk = FDK(data_reduced, ig_half)
recon = fdk.run()

islicer (recon)
# %%
show2D(recon, slice_list=('horizontal_x', 32))
# %%
