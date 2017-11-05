import scannerpy
import tensorflow as tf

class TfOpKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        # args = protobufs.TfOpArgs()
        # args.ParseFromString(config)
        # model_fn = get_model_fn(args.model_name)
        self.sess = tf.Session()
        with tf.device('/gpu:0'):
            self.image = tf.placeholder('uint8', [400, 600, 3], name="input_image")
            self.output = tf.scalar_mul(2, self.image)

    def close(self):
        pass

    def execute(self, input_columns):
        print(len(input_columns))
        scaled_input_columns = \
            map(lambda x: self.sess.run(self.output, {self.image: x}), input_columns)
        return scaled_input_columns

KERNEL = TfOpKernel
