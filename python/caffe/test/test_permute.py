import unittest
import tempfile
import os

import caffe 
import numpy as np



def python_net_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write("""name: 'pythonpermutenet' force_backward: true
        input: 'data' input_shape { dim: 3 dim: 2 dim: 2 }
        layer { type: 'Python' name: 'perml1' bottom: 'data' top: 'perm1'
          python_param { module: 'permute_layer' layer: 'PermuteLayer' } }
        layer { type: 'Python' name: 'perml2' bottom: 'perm1' top: 'perm2'
          python_param { module: 'permute_layer' layer: 'PermuteLayer' } }
        layer { type: 'Python' name: 'perml3' bottom: 'perm2' top: 'perm3'
          python_param { module: 'permute_layer' layer: 'PermuteLayer' } }
        """)
        return f.name

def cpp_net_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write("""name: 'cpppermutenet' force_backward: true
        input: 'data' input_shape { dim: 3 dim: 2 dim: 2 }
        layer { type: 'Permute' name: 'perml1' bottom: 'data' top: 'perm1'}
        layer { type: 'Permute' name: 'perml2' bottom: 'perm1' top: 'perm2'}
        layer { type: 'Permute' name: 'perml3' bottom: 'perm2' top: 'perm3'}
        """)
        return f.name

class TestPermutePythonLayer(unittest.TestCase):
    def setUp(self):
        pynet_file = python_net_file()
        self.pynet = caffe.Net(pynet_file, caffe.TRAIN)
        os.remove(pynet_file)

        cppnet_file = cpp_net_file()
        self.cppnet = caffe.Net(cppnet_file, caffe.TRAIN)
        os.remove(cppnet_file)

    def test_forward(self):
        data = self.pynet.blobs['data'].data
        dataArray = np.reshape(np.arange(data.size), data.shape)

        self.pynet.blobs['data'].data[...] = dataArray
        self.pynet.forward()
        
        self.assertTrue(np.allclose(self.pynet.blobs['perm1'].data, 
                                    self.pynet.blobs['perm3'].data))
        self.assertFalse(np.allclose(self.pynet.blobs['perm2'].data.shape, 
                                     self.pynet.blobs['perm3'].data.shape))
        self.assertTrue(np.allclose(self.pynet.blobs['perm2'].data, dataArray))

        self.cppnet.blobs['data'].data[...] = dataArray
        self.cppnet.forward()
        
        self.assertTrue(np.allclose(self.cppnet.blobs['perm1'].data, 
                                    self.cppnet.blobs['perm3'].data))
        self.assertFalse(np.allclose(self.cppnet.blobs['perm2'].data.shape, 
                                     self.cppnet.blobs['perm3'].data.shape))
        self.assertTrue(np.allclose(self.cppnet.blobs['perm2'].data, dataArray))

    def test_backward(self):
        diff = self.pynet.blobs['perm3'].diff
        diffArray = np.reshape(np.arange(diff.size), diff.shape)

        self.pynet.blobs['perm3'].diff[...] = diffArray
        self.pynet.backward()

        self.assertTrue(np.allclose(self.pynet.blobs['data'].diff, 
                                    self.pynet.blobs['perm2'].diff))
        self.assertFalse(np.allclose(self.pynet.blobs['perm2'].diff.shape, 
                                     self.pynet.blobs['perm3'].diff.shape))
        self.assertTrue(np.allclose(self.pynet.blobs['perm1'].diff, diffArray))

        self.cppnet.blobs['perm3'].diff[...] = diffArray
        self.cppnet.backward()

        self.assertTrue(np.allclose(self.cppnet.blobs['data'].diff, 
                                    self.cppnet.blobs['perm2'].diff))
        self.assertFalse(np.allclose(self.cppnet.blobs['perm2'].diff.shape, 
                                     self.cppnet.blobs['perm3'].diff.shape))
        self.assertTrue(np.allclose(self.cppnet.blobs['perm1'].diff, diffArray))

    def test_reshape(self):
        s = 4
        s2 = 5
        shape1 = (s, s2, s, s)
        shape2 = (s2, s, s, s)
        shapes = [shape1, shape2]
        self.pynet.blobs['data'].reshape(*shape1)
        self.pynet.forward()
        for i, blob in enumerate(self.pynet.blobs.itervalues()):
            self.assertTrue(np.allclose(blob.data.shape, shapes[i % 2]))

        self.cppnet.blobs['data'].reshape(*shape1)
        self.cppnet.forward()
        for i, blob in enumerate(self.cppnet.blobs.itervalues()):
            self.assertTrue(np.allclose(blob.data.shape, shapes[i % 2]))

if __name__== "__main__":
    unittest.main()
