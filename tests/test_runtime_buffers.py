import dataclasses
import unittest
import numpy as np
from runtime.buffers import SharedArrayPool,BufferFull


class BufferTests(unittest.TestCase):
    def test_leases_read_only_and_epoch_reuse(self):
        pool=SharedArrayPool(1,1024);reader=pool.reader()
        ref=pool.publish({'xyz':np.arange(18,dtype=np.float32).reshape(6,3)},readers=['gpu','safety'])
        values=reader.arrays(ref)
        np.testing.assert_array_equal(values['xyz'],np.arange(18).reshape(6,3))
        with self.assertRaises(ValueError):values['xyz'][0,0]=9
        with self.assertRaises(BufferFull):pool.publish({'a':np.ones(3)},readers=['gpu'])
        pool.release(ref,'gpu')
        with self.assertRaises(BufferFull):pool.publish({'a':np.ones(3)},readers=['gpu'])
        pool.release(ref,'safety');values.clear()
        newer=pool.publish({'a':np.ones(3)},readers=['gpu'])
        with self.assertRaises(ValueError):reader.arrays(ref)
        with self.assertRaises(ValueError):pool.release(ref,'gpu')
        pool.release(newer,'gpu');reader.close();pool.close()

    def test_capacity_and_mutated_descriptor_rejected(self):
        pool=SharedArrayPool(1,256);reader=pool.reader()
        with self.assertRaises(BufferFull):pool.publish({'big':np.ones(200)},readers=['x'])
        ref=pool.publish({'x':np.ones(4)},readers=['x'])
        bad=dataclasses.replace(ref,content_sha256='wrong')
        with self.assertRaises(ValueError):reader.arrays(bad)
        with self.assertRaises(RuntimeError):pool.close()
        pool.release(ref,'x');reader.close();pool.close()
