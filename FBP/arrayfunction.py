import numpy as np
import pysnooper
#https://towardsdatascience.com/wrapping-numpys-arrays-971e015e14bb


HANDLED_FUNCTIONS = {}

def implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator

class MyArray(object):
    # https://github.com/cupy/cupy/blob/32014aacdb347cf3f22a186cd6e30421bb4a2fe7/cupy/_core/core.pyx#L1581
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, MyArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


    def __init__(self, ndarray):
        # self.list = np.asarray(ndarray)
        self.list = ndarray[:]
    
    
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
    # @property
    # def __array_interface__(self):
    #      return {'shape': self.shape, 
    #              'typestr': self.dtype.str,
    #              'data': (id(self.list), False),
    #              'version': 3}
    # def __array__(self, dtype=None):
    #     '''Returns either a new reference to self if dtype is not given or a new array of provided data type 
    #        if dtype is different from the current dtype of the array.'''
    #     if dtype is None:
    #         # a = self
    #         # return a
    #         ret = [el for el in self.list]
    #         return MyArray(ret)
    #     if dtype != self.dtype:
    #         ret = [dtype(el) for el in self.list]
    #         return MyArray(ret)
            
    # def len(self):
    #     return self.shape
    
    def __getitem__(self, x):
        return self.list[x]
    def __setitem__(self, key, value):
        self.list[key] = value

    def __str__(self):
        return "MyArray " + str(self.list)
    

    # apparently calling np.add requires the operator
    # __add__ to be defined otherwise there is a 
    # TypeError: unsupported operand type(s) for +: 'MyArray' and 'int'
    # complaining that we don't know how to make + 
    # however I am calling np.add not + and @implements(np.add)
    # is decorating the method below, so why is this at all required??
    def __add__(self, other):
        return self.add(other)

    @pysnooper.snoop()
    @implements(np.add)
    def add(self, *args, **kwargs):
        print ("MyClass add, out", kwargs.get('out', None), len(args))
        out = kwargs.get('out', None)
        if out is None:
            return MyArray([el + args[0] for el in self.list])
        else:
            for i,el in enumerate(self.list):
                out[i] = args[0] + el

    # implements np.sum is required when one wants to use the np.sum on this object            
    @implements(np.sum)
    def sum(self, *args, **kwargs):
        return sum(self.list)

# @pysnooper.snoop()
def main():

    A = MyArray([1,2])
    
    # test np.sum
    print ("sum" , np.sum(A))
    
    # test add
    B = A.add(2)
    
    out = MyArray([20,30])
    A.add(2,out=out)
    printit(out,'out')


    # test np.add
    # see comments on __add__ 
    B = np.add(A,2)

    print (B)
    
    out = MyArray([20,30])
    arr = np.array(out)
    printit(arr, "arr")
    print ("What is arr", type(arr), type(out), arr)
    printit(out, "out")
    
    out[0] = 100
    printit(arr, "arr")
    printit(out, "out")

    out[0] = 0
    printit(arr, "arr")
    printit(out, "out")

    # if the out parameter contains a MyArray this call will fail with
    # TypeError: return arrays must be of ArrayType
    np.add(A,2,out=out)
    # having created a numpy array from MyArray, this would mean that the MyArray is an
    # array like object, no?
    # Anyway, the out and arr variables seem to point to the same memory area: modifying out 
    # results in a modification of arr
    # this line works and the correct output is calculated and stored in arr, however it is not 
    # anymore linked to out.
    np.add(A,2,out=arr)
    
    print (arr.shape, arr.ndim, arr.dtype,  arr.__array_interface__['data'])
    printit(arr, "arr")
    printit(out, "out")

    out[0] = -1
    printit(arr, "arr")
    printit(out, "out")
    
    

def printit(obj, string):
    print(string, type(obj), obj, id(obj))
if __name__ == "__main__":
    main()