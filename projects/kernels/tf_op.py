import scannerpy
from run_nn_models.model_runners import TensorflowModelRunner

class TfOpKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        args = protobufs.TfOpArgs()
        args.ParseFromString(config)
        self.model_runner = TensorflowModelRunner(args.model_name,
                                                  args.batch_size)

    def close(self):
        pass

    def execute(self, input_columns):
        return self.model_runner.execute(input_columns)

KERNEL = TfOpKernel
