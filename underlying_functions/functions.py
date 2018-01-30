import numpy as np
from scipy import stats

def generate_xy(func, xrange,num=500):
    """
    Returns the x and the y axis to be plot
    :param func: function to be evaluated
    :param xrange: max and min of the range
    :param num: number of points to evaluate
    :return:
    """
    xmin = np.min(xrange)
    xmax = np.max(xrange)
    x_axis = np.linspace(xmin,xmax, num=num)
    y_axis = np.apply_along_axis(np.vectorize(func),0,x_axis)
    return x_axis,y_axis

def step1(x):
    p=[]
    if x < 0.2:
        p = 0.2
    if x >= 0.2 and x < 0.3:
        p = 0.8
    if x >= 0.3:
        p = 0.1
    return p

def complex_trig(x):
    p = 1 / (12 * (np.sin(13 * x) * np.sin(27 * x) + 1))
    return p

def uniform_50(x):
    return 0.5

def linear(x):
    return 0.1 + (x-0.1)*0.5

def triangle30(x):
    p=[]
    if x < 0.3:
        p = 0.1 + x
    else:
        p = 0.4 - 0.25*x
    return p

def binormal4080(x):
    norm1 = stats.norm(loc=0.4,scale=0.05)
    norm2 = stats.norm(loc=0.8, scale=0.05)
    p = (norm1.pdf(x)/norm1.pdf(0.4) + norm2.pdf(x)/norm2.pdf(0.8))
    return p

def normal80(x):
    norm = stats.norm(loc=0.8,scale=0.5)
    p = norm.pdf(x)/norm.pdf(0.8)
    return p