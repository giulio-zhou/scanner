import numpy as np
import scannerpy
import tensorflow as tf
from tf_models import get_model_fn

def get_tensors_by_name(graph, tensor_names):
    return \
        [graph.get_tensor_by_name(tensor_name) for tensor_name in tensor_names]

class TfOpKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        args = protobufs.TfOpArgs()
        args.ParseFromString(config)
        self.batch_size = args.batch_size
        self.model_dict = get_model_fn(args.model_name)

        checkpoint_path = self.model_dict['checkpoint_path']
        input_tensors = self.model_dict['input_tensors']
        output_tensors = self.model_dict['output_tensors']

        # Load checkpoint
        self.tf_graph = tf.Graph()
        tf_graph = self.tf_graph
        with tf_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(checkpoint_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # TODO: Handle batch size.

        # Set up session with appropriate parameters.
        with tf_graph.as_default():
            self.sess = tf.Session(graph=tf_graph)
            with tf.device('/gpu:0'):
                # self.image = tf.placeholder('uint8', [self.batch_size, 400, 600, 3], name="input_image")
                self.input_tensors = \
                    get_tensors_by_name(tf_graph, input_tensors)
                self.output_tensors = \
                    get_tensors_by_name(tf_graph, output_tensors)

    def close(self):
        pass

    def execute(self, input_columns):
        """
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
        """
        output_pp_fns = self.model_dict['output_processing_fns']
        feed_dict = \
            self.model_dict['session_feed_dict_fn'](self.input_tensors, cols)
        outputs = self.sess.run(self.output_tensors, feed_dict)
        post_processed_outputs = \
            map(lambda fn, x: fn(x), zip(output_pp_fns, outputs))
        return [[x] for x in post_processed_outputs]

KERNEL = TfOpKernel
