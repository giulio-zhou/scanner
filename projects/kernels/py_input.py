import numpy as np
import scannerpy

class PyInputKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs

    def close(self):
        pass

    def execute(self, input_columns):
        print('PyInputKernel - len=%d\n' % len(input_columns))
        return input_columns

KERNEL = PyInputKernel
