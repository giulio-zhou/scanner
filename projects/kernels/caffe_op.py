import caffe
import numpy as np
import scannerpy
import tensorflow as tf
from caffe_models import get_model_fn

class CaffeOpKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        args = protobufs.TfOpArgs()
        args.ParseFromString(config)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.batch_size = args.batch_size
        self.model_dict = get_model_fn(args.model_name,
                                       batch_size=self.batch_size)

        caffe.set_mode_gpu()
        caffe.set_device(0)

        self.image_width, self.image_height = self.model_dict['input_dims']
        self.model_path = self.model_dict['model_prototxt_path']
        self.weights_path = self.model_dict['model_weights_path']

        self.model = caffe.Net(self.model_path, self.weights_path, caffe.TEST)
        self.model.blobs['data'].reshape(
            self.batch_size, 3, self.image_height, self.image_width)

    def close(self):
        pass

    def execute(self, input_columns):
        inputs = \
            self.model_dict['input_preprocess_fn'](self.sess, input_columns)
        outputs = self.model_dict['inference_fn'](self.model, inputs)
        post_processed_outputs = self.model_dict['post_processing_fn'](
            input_columns, outputs, self.sess)
        return post_processed_outputs

KERNEL = CaffeOpKernel
