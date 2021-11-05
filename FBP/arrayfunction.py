import numpy as np

HANDLED_FUNCTIONS = {}

def implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator

class MyArray:
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, MyArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    def __init__(self, dlist):
        self.list = dlist[:]

    @implements(np.add)
    def add(self, *args, **kwargs):
        if not 'out' in kwargs.keys():
            return MyArray([el + args[0] for el in self.list])
        else:
            for el in self.list:
                el += args[0]

if __name__ == "__main__":
    A = MyArray([1,2])
    B = np.add(A,2)
    print (B.list)

