#%%
import zarr
import os
import paramiko as ssh
import numpy as np
from cil.utilities.display import show2D
import time

private_key = os.path.abspath('C:/Users/ofn77899/.ssh/id_rsa')
pkey = ssh.RSAKey.from_private_key_file(private_key,password=None)
conn = {'host':'vishighmem01.esc.rl.ac.uk', 'username':'edo', 'port':22, 'pkey': pkey }


# small test file
g = zarr.open_group('sftp:///mnt/data/edo/Data/PSDI/data/group.zarr', storage_options=conn)
#%%
rzdata = np.asarray(g['/bar'])

print (rzdata)
# %%
# CT dataset
ct = zarr.open_group('sftp:///mnt/data/edo/Dev/CIL-work/PSDI/scanner_data_chunks_32cube.zarr', 
                      storage_options=conn)
# ct = zarr.open_group('sftp:///mnt/data/edo/Dev/CIL-work/PSDI/scanner_data_chunks.zarr', 
#                       storage_options=conn)
# %%


#%%

fsmap = zarr.storage.FSStore('sftp:///mnt/data/edo/Data/PSDI/scanner_data_chunks.zarr', **conn)
print (list(fsmap))

#%%
fsmap = zarr.storage.FSStore('sftp:///mnt/data/edo/Dev/CIL-work/PSDI/scanner_data_chunks_32cube.zarr',
     **conn)
print (list(fsmap))


#%%
t0 = time.time()
slice = ct['/tomodata/array'][100]
t1 = time.time()

show2D(slice, title="Slice transfer time {:0.3f}s".format(t1-t0))
# %%
# access an image on the non chunking axis
t0 = time.time()
slice = ct['/tomodata/array'][:,100,:]
t1 = time.time()
show2D(slice, title="Reslice transfer time {:0.3f}s".format(t1-t0))
# %%
i,j,k = (32*0,32*0,32*0)
nx, ny, nz = (1, 5, 5)
t0 = time.time()
slice = ct['/tomodata/array'][i:i+nx*32,j:j+ny*32,j:j+nz*32]
t1 = time.time()
show2D([slice[0,:,:],slice[19,:,:]], 
       title=["ROI ({}x{}x{}) transfer time {:0.3f}s"\
           .format(nx*32, ny*32, nz*32, t1-t0) for _ in range(2)])
#%%
show2D(slice, slice_list=(2,20), 
       title="ROI ({}x{}x{}) transfer time {:0.3f}s"\
           .format(nx*32, ny*32, nz*32, t1-t0) )
# %%
