import dask
from dask import delayed
import numpy as np


def add(array, scalar):
    ret = []
    array += scalar
    ret.append( id(array) )
    return ret

if __name__ == '__main__':
    N = 2
    a = [np.zeros((1024,), dtype=np.float32) for _ in range(N)]


    max_proc = len(a)
    num_parallel_processes = 6
    i = 0
    while (i < max_proc):
        j = 0
        while j < num_parallel_processes:
            if j + i == max_proc:
                break
            j += 1
        # set up j delayed computation
        procs = []
        for idx in range(j):
            print ("append job ", idx)
            projection = a[idx]
            procs.append(delayed( add )(projection, np.pi))
        # call compute on j (< num_parallel_processes) processes concurrently
        # this limits the amount of memory required to store the output of the 
        # phase retrieval to j projections.
        res = dask.compute(*procs[:j])
        
        i += j

    print (len(res), len(a))
    for i, el in enumerate(res):
        print ( "Orig id {} \nDask id {}".format(id(a[i]), el[0]))
        print (a[i][0])
