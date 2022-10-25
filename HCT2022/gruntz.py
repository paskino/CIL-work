#%%

import numpy as np

a = np.asarray([
    [0.7, 4.0],
    [3.3, 4.7],
    [5.6, 4.0],
    [7.5, 1.3], 
    [6.4, -1.1], 
    [4.4, -3.0],
    [0.3, -2.5], 
    [-1.1,1.3],
    [3,1] # added to Gruntz dataset
])

x, y = a.T
# %%


def fit_circle_n(data):
    '''Circle fitting by linear and nonlinear least squares in N Dimensions
    
    Parameters
    ----------
    data : array with the coordinates of the data. Each row is a different data point.

    Returns
    -------
    coordinates of the centre
    r : radius of the circle
    
    References
    ----------
    
    Journal of Optimisation Theory and Applications
    https://link.springer.com/article/10.1007/BF00939613
    From https://core.ac.uk/download/pdf/35472611.pdf
    '''
    B = np.vstack((data.T, np.ones(len(data))))
    d = np.sum(np.multiply(data,data), axis=1)

    res = np.linalg.lstsq(B.T,d, rcond=None)
    y = res[0]
    val = 0
    centre = np.zeros((len(res[0])-1))
    for i,el in enumerate(res[0][:-1]):
        centre[i] = el * 0.5 
        val += 0.25 * el * el

    r = np.sqrt(val + y[-1])

    return (centre,r)

def fit_circle(x,y):
    '''Circle fitting by linear and nonlinear least squares in 2D
    
    Parameters
    ----------
    x : array with the x coordinates of the data
    y : array with the y coordinates of the data. It has to have the
        same length of x.

    Returns
    -------
    x0 : x coordinate of the centre
    y0 : y coordinate of the centre
    r : radius of the circle
    
    References
    ----------

    Journal of Optimisation Theory and Applications
    https://link.springer.com/article/10.1007/BF00939613
    From https://core.ac.uk/download/pdf/35472611.pdf
    '''
    if len(x) != len(y):
        raise ValueError('X and Y array are of different length')
    data = np.vstack((x,y))

    B = np.vstack((data, np.ones(len(x))))
    d = np.sum(np.multiply(data,data), axis=0)

    res = np.linalg.lstsq(B.T,d, rcond=None)
    y = res[0]
    x0 = y[0] * 0.5 
    y0 = y[1] * 0.5
    r = np.sqrt(x0**2 + y0**2 + y[2])

    return (x0,y0,r)

# %%

res = fit_circle_n(a)
print (res)
x0,y0,r = fit_circle(x,y)
print (x0,y0,r)
# %%
