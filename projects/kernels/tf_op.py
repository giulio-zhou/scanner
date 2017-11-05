import numpy as np
import scannerpy
import tensorflow as tf

class TfOpKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        args = protobufs.TfOpArgs()
        args.ParseFromString(config)
        # model_fn = get_model_fn(args.model_name)
        self.batch_size = args.batch_size
        self.sess = tf.Session()
        with tf.device('/gpu:0'):
            self.image = tf.placeholder('uint8', [self.batch_size, 400, 600, 3], name="input_image")
            self.output = tf.scalar_mul(2, self.image)

    def close(self):
        pass

    def execute(self, input_columns):
        input_columns = input_columns[0]
        num_entries = len(input_columns)
        # Pad input if necessary
        if num_entries < self.batch_size:
            padding = [input_columns[0]] * (self.batch_size - num_entries)
            inputs = np.array(input_columns + padding)
        else:
            inputs = np.array(input_columns)

        scaled_input_columns = self.sess.run(self.output, {self.image: inputs})
        return [[scaled_input_columns[i] for i in range(num_entries)]]

KERNEL = TfOpKernel
