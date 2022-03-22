#%%
# https://github.com/TomographicImaging/CIL/issues/1105

from cil.io import TXRMDataReader, NEXUSDataReader, NEXUSDataWriter
import os
import numpy as np
import logging
import timeit
import time


results = []

logging.basicConfig(level=logging.INFO)
assert_correct_values = False

repeat = 3

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
    print(do_hdf5)
#%%
    res = []
    results.append(res)
    
    no_compression_nxs = 'scanner_data.nxs'
    writer = NEXUSDataWriter(data=scanner_data, file_name=no_compression_nxs)
    logging.info(f"writing to nexus")
    # writer.write()
    t = timeit.timeit('writer.write()', globals={'writer':writer}, number=repeat)/repeat
    logging.info(f"Writing to nexus with default NEXUSDataWriter took {t}s")
    
#%%
    action = f"loading from NEXUSDataReader default nexus {writer.file_name}"
    logging.info(action)
    for i in range(repeat):
        t0 = time.time()
        dchunk = NEXUSDataReader(file_name=no_compression_nxs).read()
        t1 = time.time()
        # average time
        logging.info("{} took {}s".format(action, t1-t0))
        dt = dt * (i/(i+1)) + (t1-t0) / (i+1)
        
    logging.info(f"{action} took average {dt}s")
    # Chunking	Compression	Compression opts	Bit shuffling	Final size	Write time	Read time	Repeats	Compression ratio
    res = ['HDF5 no compression', 0, 0, 0, 0, os.stat(no_compression_nxs).st_size, t, dt, 1]
    results.append(res)
    
#%%
    # compressed Nexus
    h5tests = [ [ 'gzip', 4, True ],[ 'lzf', None, True ] ]
    for compression_algo, compression_level, shuffle in h5tests:
        writer = NEXUSDataWriter(data=scanner_data, 
                                file_name=os.path.abspath('scanner_data_chunk.nxs'))
        # shuffle = True
        # compression_level = 4
        # compression_algo = None
        writer.h5compress = [compression_algo, compression_level, shuffle]
        
        logging.info(f"Writing to nexus with chunk {chunks} and compression {writer.h5compress}")
        writer.chunks = tuple(chunks)

        t = timeit.timeit('writer.write()', globals={'writer':writer}, number=repeat) / repeat
        # writer.write()
        
        logging.info(f"Writing to nexus with chunk {chunks} and compression {writer.h5compress} took {t}s")
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
        action = f"loading from chunked nexus {writer.file_name}"
        logging.info(action)
        for i in range(repeat):
            t0 = time.time()
            dchunk = NEXUSDataReader(file_name=writer.file_name).read()
            t1 = time.time()
            # average time
            logging.info("{} took {}s".format(action, t1-t0))
            dt = dt * (i/(i+1)) + (t1-t0) / (i+1)
        
        logging.info(f"loading from chunked/compressed nexus {writer.file_name} took {dt}s")

        # Chunking	Compression	Compression opts	Bit shuffling	Final size	Write time	Read time	Repeats	Compression ratio
        res = ['HDF5 compression', 1, compression_algo, compression_level, shuffle, os.stat(writer.file_name).st_size, t, dt, stat[0]/stat[1]]
        results.append(res)

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
    logging.info("Nexus compression ratio {} {}".format(stat[0]/stat[1], stat))

#%%    
    

#%% 
# testing Zarr
import zarr
from numcodecs import Blosc
logging.info("Test Zarr")
logging.info("Use Blosc compressor")
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

#z = zarr.open('data/example.zarr', mode='w', shape=1000000, dtype='i4')
# root = zarr.group()


logging.info("Create dataset with shape {}".format(scanner_data.shape))

arr = None
# t = timeit.timeit('arr = zdata.create_dataset("array", data=scanner_data.as_array(), shape=scanner_data.shape, chunks=chunks, dtype=scanner_data.dtype, compressor=compressor)', 
#          globals={'zdata':zdata, 'scanner_data':scanner_data, 
#           'chunks': chunks, 'compressor': compressor, 'arr': arr}, number=1)
action = 'Zarr '
for i in range(repeat):
    zfname = 'scanner_data_chunks.zarr'
    logging.info("Open for write {}".format(zfname))
    store = zarr.DirectoryStore(zfname)
    root = zarr.group(store=store, overwrite=True)
    logging.info("create groups".format(zfname))
    zdata = root.create_group('/tomodata')
    # zmeta = root.create_group('/tomometa')
    t0 = time.time()
    arr = zdata.create_dataset('array', data=scanner_data.as_array(), 
                                shape=scanner_data.shape, chunks=chunks, 
                                dtype=scanner_data.dtype, compressor=compressor)
    t1 = time.time()
    # average time
    logging.info("{} took {}s".format(action, t1-t0))
    t = t * (i/(i+1)) + (t1-t0) / (i+1)

    logging.info("Close store")
    store.close()
    

print (arr.info)


logging.info(f"Writing to Zarr with chunk {chunks} and compression {compressor} took {dt}s")

#%%
logging.info("Read file")
action = "Reading Zarr in memory"
for i in range(repeat):
    zfname = 'scanner_data_chunks.zarr'
    store = zarr.DirectoryStore(zfname)
    root = zarr.group(store=store)

    t0 = time.time()
    rzdata = np.asarray(root['/tomodata/array'])
    t1 = time.time()
    # average time
    logging.info("{} took {}s".format(action, t1-t0))
    dt = dt * (i/(i+1)) + (t1-t0) / (i+1)


# print (rzdata)
logging.info("Reading Zarr in memory took {}s".format(dt))

# Chunking	Compression	Compression opts	Bit shuffling	Final size	Write time	Read time	Repeats	Compression ratio

# compression ratio is written in arr.info
res = ['Zarr compression', 1, compressor.cname, compressor.clevel, compressor.shuffle, 
        os.stat(zfname).st_size, t, dt, arr.nbytes/arr.nbytes_stored]
results.append(res)
#%%
if assert_correct_values:
    np.testing.assert_allclose(rzdata, scanner_data.as_array())


with open("results.csv", "w") as f:
    import csv
    w = csv.writer(f, dialect='excel')
    for row in results:
        w.writerow(row)

logging.info("All OK.")


# %%
