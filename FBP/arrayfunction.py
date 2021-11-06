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
            print ("here")
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

    @implements(np.add)
    def __add__(self, *args, **kwargs):
        if 'out' not in kwargs.keys():
            return MyArray([el + args[0] for el in self.list])
        else:
            for el,ol in zip(self.list, out.list):
                ol = args[0] + el
    @implements(np.sum)
    def sum(self, *args, **kwargs):
        return sum(self.list)

if __name__ == "__main__":
    A = MyArray([1,2])
    
    print (np.sum(A))
    
    out = MyArray([0,0])
    B = np.add(A,2)
    print (B.list)
    np.add(A,2, out=out)
    print (out.list)
    

