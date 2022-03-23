#%%
import zarr
import os
import paramiko as ssh
import numpy as np


private_key = os.path.abspath('C:/Users/ofn77899/.ssh/id_rsa')
pkey = ssh.RSAKey.from_private_key_file(private_key,password=None)
conn = {'host':'vishighmem01.esc.rl.ac.uk', 'username':'edo', 'port':22, 'pkey': pkey }


g = zarr.open_group('sftp:///mnt/data/edo/Data/PSDI/data/group.zarr', storage_options=conn)
#%%
rzdata = np.asarray(g['/bar'])

print (rzdata)