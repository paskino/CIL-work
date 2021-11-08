import numpy as np

#https://towardsdatascience.com/wrapping-numpys-arrays-971e015e14bb


HANDLED_FUNCTIONS = {}

def implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator

class MyArray(object):
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, MyArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    def __init__(self, dlist):
        self.list = dlist[:]
    @property
    def ndim(self):
        return 1
    @property
    def shape(self):
        return (len(self.list),)
    @property
    def dtype(self):
        return np.int64
    # https://stackoverflow.com/questions/40378427/numpy-formal-definition-of-array-like-objects
    # def __array__(self, dtype=None):
    #     if dtype is None:
    #         dtype = self.dtype
    #     if dtype != self.dtype:
    #         return np.zeros(self.shape, dtype)
    #     else:
    #         ret = np.empty(self.shape, dtype=self.dtype)
    #         for i, el in enumerate(self.list):
    #             ret[i] = el
    #         return ret
    # def len(self):
    #     return self.shape
    
    # def __getitem__(self, x):
    #     return self.list[x]

    @implements(np.add)
    def __add__(self, *args, **kwargs):
        print ("here")
        out = kwargs.get('out', None)
        if out is None:
            return MyArray([el + args[0] for el in self.list])
        else:
            for i in enumerate(self.list):
                out[i] = args[0] + self.list[i]
            
    @implements(np.sum)
    def sum(self, *args, **kwargs):
        return sum(self.list)

if __name__ == "__main__":
    A = MyArray([1,2])
    a = np.asarray(A)
    print (a)
    print (np.sum(A))
    
    out = MyArray([0,0])

    B = np.add(A,2)
    print (B.list)
    np.add(A,2,out=out)
    print (out.list)
    

