import numpy as np
import math
import csv
from matplotlib import pyplot as plt

def distance_err(prediction, label):
    distance = np.sqrt(np.sum(np.square(prediction-label),1))
    return distance


def mean_err_distance(d):
    med = np.mean(d)
    return med


def var_distance_err(d):
    vde = np.var(d)
    return (vde)


def max_err_distance(d):
    maxed = np.max(d)
    return (maxed)


def cdf_figure(d):
    hist, b = np.histogram(d, bins=np.arange(0,15,1))
    num = hist.shape[0]

    h1 = b
    h0 = np.zeros([num+1])
    for i in range(0, num):
        if i == 0:
            h0[i+1] = hist[i]
        else:
            h0[i+1] = (h0[i] + hist[i])

    plt.figure()
    plt.xlabel('Distance Error')
    plt.ylabel('Probability')
    plt.title('CDF in a offical environment')

    plt.xlim(0,15,1)
    plt.ylim(0,1.1,0.1)

    plt.plot(h1, h0 / d.shape[0],'r-*')
    plt.show()


def cdf(d):
    hist, b = np.histogram(d, bins=np.arange(0,15,1))
    num = hist.shape[0]

    h1 = b
    h0 = np.zeros([num+1])
    for i in range(0, num):
        if i == 0:
            h0[i+1] = hist[i]
        else:
            h0[i+1] = (h0[i] + hist[i])
    return h1, h0 / d.shape[0]


def all(d):
    result = np.zeros([3])
    result[0] = mean_err_distance(d)
    result[1] = var_distance_err(d)
    result[2] = max_err_distance(d)
    #cdf_figure(d)

    return result




