import tensorflow as tf

def identity(x):
    return x

def np_to_string(x):
    return x.tostring()

def get_full_checkpoint_path(checkpoint_dir):
    return 'tf_nets/%s/frozen_inference_graph.pb' % checkpoint_dir

def ssd_mobilenet_v1_coco():
    return {
        'checkpoint_path': get_full_checkpoint_path('ssd_mobilenet_v1_coco'),
        'input_tensors': ['image_tensor:0'],
        'output_tensors': ['detection_boxes:0', 'detection_scores:0',
                           'detection_classes:0', 'num_detections:0'],
        'output_processing_fns': [np_to_string, np_to_string,
                                  np_to_string, np_to_string],
        'session_feed_dict_fn': \
            lambda input_tensors, cols: {input_tensors[0]: cols[0]}
    }

# This should return a dictionary with the following items:
#     "checkpoint_path": directory containing frozen_inference_graph.pb
#     "input_tensors": list of names of input tensors
#     "output_tensors": list of names of output tensors
#     "output_processing_fns": list of output processing functions
#     "session_feed_dict_fn": function that generates feed_dict given \
#                             input_tensors and input_cols
def get_model_fn(model_name):
    if model_name == 'ssd_mobilenet_v1_coco':
        return ssd_mobilenet_v1_coco()
    else:
        raise Exception("Could not find network with name %s" % model_name)
