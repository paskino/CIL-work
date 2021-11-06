import numpy as np

#https://towardsdatascience.com/wrapping-numpys-arrays-971e015e14bb


import functools
# FIRST
HANDLED_FUNCTIONS = {}

class Physical():
    
    def __init__(self, value, unit=""):
        self._value = value
        self._unit = unit
    
    # ... other methods here, see above
       
    # SECOND
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects
        if not all(issubclass(t, Physical) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    
# THIRD
def implements(numpy_function):
    """Register an __array_function__ implementation for Physical objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator
    
# FOURTH
@implements(np.mean)
def np_mean_for_physical(x, *args, **kwargs):
    print ("np_mean_for_physical")
    # first compute the numerical value, with no notion of unit
    mean_value = np.mean(x._value, *args, **kwargs)
    # construct a Physical instance with the result, using the same unit
    return Physical(mean_value, x._unit)
 
weights = Physical(np.array([55.6, 45.7, 80.3]), "kilogram")
heights = Physical(np.array([1.64, 1.85, 1.77]), "meter")
# bmi = weights/heights**2

print(np.mean(weights)._value, np.mean(weights._value)) # 19.885411834844252 kilogram/meter^2



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
    # def __add__(self, other):
    #     return np.add(self, other)
if __name__ == "__main__":
    A = MyArray([1,2])
    
    print (np.sum(A))
    
    out = MyArray([0,0])
    B = np.add(A,2)
    print (B.list)
    np.add(A,2, out=out)
    print (out.list)
    

