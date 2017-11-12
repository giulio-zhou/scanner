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
        self.model_dict = get_model_fn(args.model_name,
                                       batch_size=self.batch_size)

        mode = self.model_dict['mode']
        checkpoint_path = self.model_dict['checkpoint_path']
        input_tensors = self.model_dict['input_tensors']
        output_tensors = self.model_dict['output_tensors']
        sess_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)

        # NOTE: Batching is only allowed in python mode because frozen graphs
        #       cannot be modified.

        with tf.device('/gpu:0'):
            if mode == 'frozen_graph':
                tf_graph = tf.Graph()
                with tf_graph.as_default():
                    od_graph_def = tf.GraphDef()
                    with tf.gfile.GFile(checkpoint_path, 'rb') as fid:
                        serialized_graph = fid.read()
                        od_graph_def.ParseFromString(serialized_graph)
                        tf.import_graph_def(od_graph_def, name='')
                self.sess = tf.Session(graph=tf_graph, config=sess_config)
            elif mode == 'python':
                self.sess = tf.Session(config=sess_config)
                self.model_dict['model_init_fn']()
                saver = tf.train.Saver()
                saver.restore(self.sess, checkpoint_path)
                tf_graph = self.sess.graph
            elif mode == 'keras':
                self.sess = tf.Session(config=sess_config)
                from keras import backend as K
                K.set_session(self.sess)
                self.model_dict['model_init_fn'](K)
                tf_graph = self.sess.graph
            else:
                raise Exception("Invalid model loading mode: %s" % mode)

        self.input_tensors = get_tensors_by_name(tf_graph, input_tensors)
        self.output_tensors = get_tensors_by_name(tf_graph, output_tensors)

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
        
        feed_dict = self.model_dict['session_feed_dict_fn'](self.input_tensors,
                                                            input_columns)
        outputs = self.sess.run(self.output_tensors, feed_dict)
        post_processed_outputs = \
            self.model_dict['post_processing_fn'](input_columns, outputs)
        return post_processed_outputs

KERNEL = TfOpKernel
