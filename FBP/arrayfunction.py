import numpy as np
import logging
# import torch
# import pysnooper
#https://towardsdatascience.com/wrapping-numpys-arrays-971e015e14bb
import inspect
# initializing function
def get_function_name():
    # get the frame object of the function
    frame = inspect.currentframe()
    return frame.f_code.co_name

HANDLED_FUNCTIONS = {}

def implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator

class MyArray(object):
    # https://github.com/cupy/cupy/blob/32014aacdb347cf3f22a186cd6e30421bb4a2fe7/cupy/_core/core.pyx#L1581
    # @pysnooper.snoop()
    def __array_function__(self, func, types, args, kwargs):
        logging.debug('{} {}'.format(inspect.currentframe().f_code.co_name, func))
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, MyArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    
    # @pysnooper.snoop()
    def __array_ufunc__(self, ufunc, method, inputs, *args, **kwargs):
        logging.debug('{} {}'.format(inspect.currentframe().f_code.co_name, ufunc))
        if ufunc not in HANDLED_FUNCTIONS:
            return NotImplemented
        out = kwargs.pop('out', None)
    
        if out is not None:
            HANDLED_FUNCTIONS[ufunc](inputs, *args, out=out[0], **kwargs)
            return
        else:
            return HANDLED_FUNCTIONS[ufunc](inputs, *args, out=None, **kwargs)
        
    def __init__(self, ndarray):
        # self.list = np.asarray(ndarray)
        self.list = ndarray[:]
        # self.__array_ufunc__=None
    
    
    @property
    def ndim(self):
        return 1
        # return self.list.ndim
    
    @property
    def shape(self):
        return (len(self.list), )
    
    @property
    def dtype(self):
        return np.dtype(np.int32)
        
    # https://stackoverflow.com/questions/40378427/numpy-formal-definition-of-array-like-objects
    
    def __getitem__(self, x):
        return self.list[x]
    def __setitem__(self, key, value):
        self.list[key] = value

    def __str__(self):
        return "MyArray " + str(self.list)
    
    def __add__(self, other, *args, **kwargs):
        logging.debug('{}'.format(inspect.currentframe().f_code.co_name))
        return self.add(other, *args, **kwargs)

    # @pysnooper.snoop()
    @implements(np.add)
    def add(self, *args, **kwargs):
        strng = "MyClass add, out {} {}".format( kwargs.get('out', None), len(args) )
        logging.debug('{} {}'.format(inspect.currentframe().f_code.co_name, strng))
        # print('Sto chiamando la np.add dentro implements')
        out = kwargs.get('out', None)
        if out is None:
            return MyArray([el + args[0] for el in self.list])
        else:
            for i,el in enumerate(self.list):
                out[i] = args[0] + el

    # implements np.sum is required when one wants to use the np.sum on this object            
    @implements(np.sum)
    def sum(self, *args, **kwargs):
        return sum(self.list)  # return self.list.ndim
    
# @pysnooper.snoop()
def main():
    logging.basicConfig(level=logging.DEBUG)

    A = MyArray(np.array([1,2]))
    
    # test np.sum
    print ("sum" , np.sum(A, axis=1))
    
    # test add
    B = A.add(2)
    printit(B, 'B')
    
    out = MyArray([20,30])
    printit(out,'out')
    A.add(2,out=out)
    printit(out,'out')

    # test np.add
    # see comments on __add__ 
    #B = A+2
    B = np.add(A,2)
    printit(B, 'B')

    print (">>>>>>>>>>>>>")
    B = A+2
    printit(B, 'B')
    print (">>>>>>>>>>>>>")

    # out = MyArray([20,30])
    np.add(A,2,out=out)
    printit(out, "out")
    

    exit(0)

    arr = np.array(out)
    printit(arr, "arr")
    print ("What is arr", type(arr), type(out), arr)
    printit(out, "out")
    
    out[0] = 100
    printit(arr, "arr")
    printit(out, "out")

    # out[0] = 0
    # printit(arr, "arr")
    # printit(out, "out")

    # # if the out parameter contains a MyArray this call will fail with
    # # TypeError: return arrays must be of ArrayType
    # np.add(A,2,out=out)
    # # having created a numpy array from MyArray, this would mean that the MyArray is an
    # # array like object, no?
    # # Anyway, the out and arr variables seem to point to the same memory area: modifying out 
    # # results in a modification of arr
    # # this line works and the correct output is calculated and stored in arr, however it is not 
    # # anymore linked to out.
    np.add(A,2,out=arr)
    
    # print (arr.shape, arr.ndim, arr.dtype,  arr.__array_interface__['data'])
    printit(arr, "arr")
    printit(out, "out")

    # out[0] = -1
    # printit(arr, "arr")
    # printit(out, "out")
    
    

def printit(obj, string):
    print(string, type(obj), obj, id(obj))
if __name__ == "__main__":
    main()