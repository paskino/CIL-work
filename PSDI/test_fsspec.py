#%%
import zarr
import os
import paramiko as ssh
import numpy as np


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
ct = zarr.open_group('sftp:///mnt/data/edo/Data/PSDI/scanner_data_bichunks.zarr', 
                      storage_options=conn)

# %%

from cil.utilities.display import show2D
import time
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
