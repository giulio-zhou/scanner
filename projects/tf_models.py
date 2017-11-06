<<<<<<< HEAD
import numpy as np
import tensorflow as tf
import utils.visualization_utils as vis_util
from skimage import img_as_ubyte
=======
import tensorflow as tf
>>>>>>> e79288eaebff74774071f76cba8777e8a79d1281

def identity(x):
    return x

def np_to_string(x):
    return x.tostring()

def get_full_checkpoint_path(checkpoint_dir):
    # base_dir = 'tf_nets/%s/' % checkpoint_dir
    # return {'meta_graph': base_dir + 'model.ckpt.meta',
    #         'checkpoint': base_dir + 'model.ckpt.data-00000-of-00001'}
    return 'tf_nets/%s/frozen_inference_graph.pb' % checkpoint_dir

def ssd_mobilenet_v1_coco():
    def post_process_fn(inputs, outputs):
        image_np = inputs[0][0]
        boxes, scores, classes, num_detections = outputs
        category_index = \
            {i: {'name': 'stuff', 'id': i, 'display_name': 'stuff'} for i in range(90)}
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        image_np = img_as_ubyte(image_np)
        return [[image_np]]

    return {
        'checkpoint_path': get_full_checkpoint_path('ssd_mobilenet_v1_coco'),
        'input_tensors': ['image_tensor:0'],
        'output_tensors': ['detection_boxes:0', 'detection_scores:0',
                           'detection_classes:0', 'num_detections:0'],
        'output_processing_fn': post_process_fn,
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
