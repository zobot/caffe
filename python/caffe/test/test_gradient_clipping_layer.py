import unittest
import tempfile
import os

import caffe
import numpy as np
from numpy.linalg import norm

class GradientClipLayer(caffe.Layer):
    """A layer that clips the gradient for each slice along the batch_axis, 
       which is currently hardcoded to 1."""

    def setup(self, bottom, top):
        self.gradient_clip = 5.0
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data

    def backward(self, top, propagate_down, bottom):
        num_batch = bottom[0].diff.shape[1]
        for i in range(num_batch):
            diff_slice = top[0].diff[:, i]
            norm_slice = norm(diff_slice)

            if norm_slice > self.gradient_clip:
                bottom[0].diff[:, i] = self.gradient_clip / norm_slice * diff_slice
            else:
                bottom[0].diff[:, i] = diff_slice

def python_net_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write("""name: 'pythonnet' force_backward: true
        input: 'data' input_shape { dim: 10 dim: 9 dim: 8 dim: 4}
        layer { type: 'Python' name: 'one' bottom: 'data' top: 'one'
          python_param { module: 'test_gradient_clipping_layer' layer: 'GradientClipLayer' } }""")
        return f.name

def cpp_net_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write("""name: 'cppnet' force_backward: true
        input: 'data' input_shape { dim: 10 dim: 9 dim: 8 dim: 4}
        layer { type: 'GradientClip' name: 'one' bottom: 'data' top: 'one'
          gradient_clip_param { batch_axis: 1 gradient_clip: 5.0 } }""")
        return f.name

class TestGradientClipLayer(unittest.TestCase):
    def setUp(self):
        pynet_file = python_net_file()
        self.pynet = caffe.Net(pynet_file, caffe.TRAIN)
        os.remove(pynet_file)
        cppnet_file = cpp_net_file()
        self.cppnet = caffe.Net(cppnet_file, caffe.TRAIN)
        os.remove(cppnet_file)

    def test_forward(self):
        x = 8
        self.pynet.blobs['data'].data[...] = x
        self.pynet.forward()
        for y in self.pynet.blobs['one'].data.flat:
            self.assertEqual(y, x)

        self.cppnet.blobs['data'].data[...] = x
        self.cppnet.forward()
        for y in self.cppnet.blobs['one'].data.flat:
            self.assertEqual(y, x)

    def test_backward(self):
        zeroArray = 5.0 * np.reshape(np.arange(10 * 8 * 4), (10, 8, 4))
        oneArray = 0.000001 * np.reshape(np.arange(10 * 8 * 4), (10, 8, 4))

        self.pynet.blobs['one'].diff[:, 0] = zeroArray
        self.pynet.blobs['one'].diff[:, 1] = oneArray
        normzero = norm(zeroArray)

        self.pynet.backward()
        self.assertTrue(np.allclose(self.pynet.blobs['data'].diff[:, 0], 5.0 / normzero * zeroArray))
        self.assertTrue(np.allclose(self.pynet.blobs['data'].diff[:, 1], oneArray))

        self.cppnet.blobs['one'].diff[:, 0] = zeroArray
        self.cppnet.blobs['one'].diff[:, 1] = oneArray
        self.cppnet.backward()
        self.assertTrue(np.allclose(self.cppnet.blobs['data'].diff[:, 0], 5.0 / normzero * zeroArray))
        self.assertTrue(np.allclose(self.cppnet.blobs['data'].diff[:, 1], oneArray))

    def test_reshape(self):
        s = 4
        self.pynet.blobs['data'].reshape(s, s, s, s)
        self.pynet.forward()
        for blob in self.pynet.blobs.itervalues():
            for d in blob.data.shape:
                self.assertEqual(s, d)

if __name__== "__main__":
    unittest.main()
