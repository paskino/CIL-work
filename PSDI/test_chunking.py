#%%
# https://github.com/TomographicImaging/CIL/issues/1105

from cil.io import TXRMDataReader, NEXUSDataReader, NEXUSDataWriter
import os
import numpy as np
import logging
import timeit
import time

logging.basicConfig(level=logging.INFO)
assert_correct_values = False

raw_data_fname = os.path.abspath("C:/Users/ofn77899/Data/walnut/valnut/valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm")
logging.info(f"loading {raw_data_fname}")

t0 = time.time()
scanner_data = TXRMDataReader(file_name = raw_data_fname ).read()
t1 = time.time()
dt = t1-t0
logging.info(f"Reading from file with TXRMDataReader took {dt}s")

chunks = list(scanner_data.shape)
chunks[0] = 1

#%%
do_hdf5 = True
if do_hdf5:
    no_compression_nxs = 'scanner_data.nxs'
    writer = NEXUSDataWriter(data=scanner_data, file_name=no_compression_nxs)
    logging.info(f"writing to nexus")
    # writer.write()
    t = timeit.timeit('writer.write()', globals={'writer':writer}, number=1)
    logging.info(f"Writing to nexus with default NEXUSDataWriter took {t}s")

    logging.info(f"loading from NEXUSDataWriter default nexus {writer.file_name}")
    t0 = time.time()
    dchunk = NEXUSDataReader(file_name=no_compression_nxs).read()
    t1 = time.time()
    dt = t1-t0
    logging.info(f"loading from NEXUSDataWriter default nexus {writer.file_name} took {dt}s")

#%%
    # compressed Nexus
    writer.file_name = os.path.abspath('scanner_data_chunk.nxs')
    shuffle = True
    compression_level = 4
    compression_algo = 'gzip'
    writer.h5compress = [compression_algo, compression_level, shuffle]
    
    logging.info(f"Writing to nexus with chunk {chunks} and compression {writer.h5compress}")
    writer.chunks = tuple(chunks)

    t = timeit.timeit('writer.write()', globals={'writer':writer}, number=1)
    # writer.write()
    logging.info(f"Writing to nexus with chunk {chunks} and compression {writer.h5compress} took {t}s")


    logging.info(f"loading from chunked nexus {writer.file_name}")
    t0 = time.time()
    dchunk = NEXUSDataReader(file_name=writer.file_name).read()
    t1 = time.time()
    dt = t1-t0
    logging.info(f"loading from chunked/compressed nexus {writer.file_name} took {dt}s")

    if assert_correct_values:
        logging.info("assert array all close")
        np.testing.assert_allclose(dchunk.as_array(), scanner_data.as_array())

#%% 
    # test compression ratio
    import os
    stat = []
    for el in [no_compression_nxs, writer.file_name]:
        stat.append(
            os.stat(el).st_size
        )
    logging.info("Nexus compression ratio {}".format(stat[0]/stat[1]))

#%%    
    

#%% 
# testing Zarr
import zarr
from numcodecs import Blosc
logging.info("Test Zarr")
logging.info("Use Blosch compressor")
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

#z = zarr.open('data/example.zarr', mode='w', shape=1000000, dtype='i4')
# root = zarr.group()
zfname = 'scanner_data_chunks.zarr'
logging.info("Open for write {}".format(zfname))
store = zarr.DirectoryStore(zfname)
root = zarr.group(store=store, overwrite=True)
logging.info("create groups".format(zfname))
zdata = root.create_group('/tomodata')
zmeta = root.create_group('/tomometa')

logging.info("Create dataset with shape {}".format(scanner_data.shape))

arr = None
# t = timeit.timeit('arr = zdata.create_dataset("array", data=scanner_data.as_array(), shape=scanner_data.shape, chunks=chunks, dtype=scanner_data.dtype, compressor=compressor)', 
#          globals={'zdata':zdata, 'scanner_data':scanner_data, 
#           'chunks': chunks, 'compressor': compressor, 'arr': arr}, number=1)
t0 = time.time()
arr = zdata.create_dataset('array', data=scanner_data.as_array(), shape=scanner_data.shape, chunks=chunks, dtype=scanner_data.dtype, compressor=compressor)
t1 = time.time()
dt = t1 - t0

logging.info("Close store")
store.close()

print (arr.info)

logging.info(f"Writing to Zarr with chunk {chunks} and compression {compressor} took {dt}s")

#%%
logging.info("Read file")
store = zarr.DirectoryStore(zfname)
root = zarr.group(store=store)

t0 = time.time()
rzdata = np.asarray(root['/tomodata/array'])
t1 = time.time()

# print (rzdata)
logging.info("Reading Zarr in memory took {}s".format(t1-t0))
#%%
if assert_correct_values:
    np.testing.assert_allclose(rzdata, scanner_data.as_array())

logging.info("All OK.")
# %%
