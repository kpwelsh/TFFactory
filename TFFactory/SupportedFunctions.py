from .DataLoader import DataLoader
import numpy as np
DL = DataLoader()


def readFile(fp, delim, nRows):
    global DL
    return DL.sampleFile(fp, nRows, caching=True)


def splitFile(Source, SegmentDelimeter, DataDelimeter, SegmentIndex, Shape):
    data = []
    for row in Source:
        data.append(list(map(float, row.split(SegmentDelimeter)
                             [SegmentIndex].split(DataDelimeter))))
    return np.array(data).reshape((*Shape))


def testAdd(a, b):
    return [a + b]


# TF ones
import tensorflow


def AdamOptimizer(loss, learningRate):
    optimizer = tensorflow.train.AdamOptimizer(learningRate)
    return optimizer.minimize(loss)


def MomentumOptimizer(loss, learningRate, momentum):
    optimizer = tensorflow.train.MomentumOptimizer(
        learningRate, momentum=momentum)
    return optimizer.minimize(loss)


def GradientDescentOptimizer(loss, learningRate):
    optimizer = tensorflow.train.GradientDescentOptimizer(learningRate)
    return optimizer.minimize(loss)


def SampleDirichlet(concentration, sampleShape, validateArgs=False, allowNanStats=True, **kwargs):
    dist = tensorflow.distributions.Dirichlet(concentration,
                                              validate_args=validateArgs,
                                              allow_nan_stats=allowNanStats)

    return dist.sample(sampleShape, **kwargs)


def GetItem(tensor, key):
    return tensor[key]
