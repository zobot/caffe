import numpy as np

import caffe

class PermuteLayer(caffe.Layer):
    """A layer that permutes the first two axes"""

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        shape = bottom[0].data.shape
        newShape = list(shape)
        newShape[0], newShape[1] = shape[1], shape[0]
        top[0].reshape(*tuple(newShape))

    def forward(self, bottom, top):
        rearrange = np.arange(len(bottom[0].data.shape))
        rearrange[0], rearrange[1] = rearrange[1], rearrange[0]
        top[0].data[...] = np.transpose(bottom[0].data, rearrange)
        #for i in range(bottom[0].data.shape[0]):
            #top[0].data[:, i] = bottom[0].data[i, :]

    def backward(self, top, propagate_down, bottom):
        rearrange = np.arange(len(bottom[0].data.shape))
        rearrange[0], rearrange[1] = rearrange[1], rearrange[0]
        print 'bot: ', bottom[0].diff.shape
        print 'top: ', top[0].diff.shape
        bottom[0].diff[...] = np.transpose(top[0].diff, rearrange)
        #for i in range(bottom[0].data.shape[0]):
            #bottom[0].diff[i, :] = top[0].diff[:, i]
