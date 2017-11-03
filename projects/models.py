import cv2
import numpy as np
import scannerpy
import tensorflow as tf
from scannerpy.stdlib import parsers

def image(bufs, protobufs):
    # print np.frombuffer(bufs[0], dtype=np.dtype(np.uint8))
    return cv2.imdecode(np.frombuffer(bufs[0], dtype=np.dtype(np.uint8)),
                        cv2.IMREAD_COLOR)

class TestKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        self.sess = tf.Session()

    def close():
        pass

    def execute(self, input_columns):
        print input_columns
        print image(input_columns, protobufs)
        return input_columns

KERNEL = TestKernel
