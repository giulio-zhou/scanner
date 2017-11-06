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

        self.i = 0

        with tf.device('/gpu:0'):
            """
            self.sess = tf.Session()
            saver = tf.train.import_meta_graph(checkpoint_path['meta_graph'])
            saver.restore(self.sess, checkpoint_path['checkpoint'])
            self.input_tensors = \
                get_tensors_by_name(tf_graph, input_tensors)
            self.output_tensors = \
                get_tensors_by_name(tf_graph, output_tensors)
            """
            
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
                # self.image = tf.placeholder('uint8', [self.batch_size, 400, 600, 3], name="input_image")
                self.input_tensors = \
                    get_tensors_by_name(tf_graph, input_tensors)
                print(self.input_tensors)
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
        
        feed_dict = self.model_dict['session_feed_dict_fn'](self.input_tensors,
                                                            input_columns)
        outputs = self.sess.run(self.output_tensors, feed_dict)
        post_processed_outputs = \
            self.model_dict['output_processing_fn'](input_columns, outputs)
        # post_processed_outputs = \
        #     [[fn(x)] for fn, x in zip(output_pp_fns, outputs)]
        print(self.i)
        self.i += 1
        return post_processed_outputs

KERNEL = TfOpKernel
