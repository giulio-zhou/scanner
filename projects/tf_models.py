import numpy as np
import tensorflow as tf
from skimage import img_as_ubyte

def identity(x):
    return x

def np_to_string(x):
    return x.tostring()

def get_checkpoint_base_dir(checkpoint_dir):
    return 'tf_nets/%s' % checkpoint_dir

def get_single_checkpoint_path(checkpoint_dir, checkpoint_name):
    full_checkpoint_path = \
        get_checkpoint_base_dir(checkpoint_dir) + '/' + checkpoint_name
    return full_checkpoint_path

def get_frozen_graph_path(checkpoint_dir):
    full_checkpoint_path = \
        get_checkpoint_base_dir(checkpoint_dir) + '/frozen_inference_graph.pb'
    return full_checkpoint_path

def input_pre_process_fn(input_columns, batch_size):
    cols = input_columns[0]
    if len(cols) < batch_size:
        padding = [cols[0]] * (batch_size - len(cols))
        inputs = np.array(cols + padding)
    else:
        inputs = np.array(cols)
    return inputs

def mobilenet_v1_224(batch_size=1):
    def create_mobilenet_model():
        from mobilenet_v1 import mobilenet_v1
        inputs = tf.placeholder('uint8', [batch_size, None, None, 3],
                                name='image_tensor')
        resized_inputs = tf.image.resize_images(inputs, [224, 224])
        mobilenet_v1(resized_inputs, num_classes=1001, is_training=False)

    def post_process_fn(input_columns, outputs):
        num_outputs = len(input_columns)
        serialize_fn = lambda x: np.ndarray.dumps(x.squeeze())
        return [[serialize_fn(outputs[0][i]) for i in range(num_outputs)]]

    return {
        'mode': 'python',
        'checkpoint_path': get_single_checkpoint_path(
            'mobilenet', 'mobilenet_v1_1.0_224.ckpt'),
        'input_tensors': ['image_tensor:0'],
        'output_tensors': ['MobilenetV1/Logits/AvgPool_1a/AvgPool:0'],
        'post_processing_fn': post_process_fn,
        'session_feed_dict_fn': lambda input_tensors, cols: \
            {input_tensors[0]: pre_process_fn(cols, batch_size)},
        'model_init_fn': create_mobilenet_model
    }

def draw_tf_bounding_boxes(image_np, boxes, scores, classes, num_detections):
    import utils.visualization_utils as vis_util
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
    return image_np

def ssd_mobilenet_v1_coco(batch_size=1):
    def post_process_fn(inputs, outputs):
        image_np = inputs[0][0]
        boxes, scores, classes, num_detections = outputs
        image_np = draw_tf_bounding_boxes(
            image_np, boxes, scores, classes, num_detections)
        image_np = img_as_ubyte(image_np)
        return [[image_np]]

    return {
        'mode': 'frozen_graph',
        'checkpoint_path': get_frozen_graph_path('ssd_mobilenet_v1_coco'),
        'input_tensors': ['image_tensor:0'],
        'output_tensors': ['detection_boxes:0', 'detection_scores:0',
                           'detection_classes:0', 'num_detections:0'],
        'post_processing_fn': post_process_fn,
        'session_feed_dict_fn': \
            lambda input_tensors, cols: {input_tensors[0]: cols[0]}
    }

def faster_rcnn_resnet101_coco(batch_size=1):
    def post_process_fn(inputs, outputs):
        image_np = inputs[0][0]
        boxes, scores, classes, num_detections = outputs
        image_np = draw_tf_bounding_boxes(
            image_np, boxes, scores, classes, num_detections)
        image_np = img_as_ubyte(image_np)
        return [[image_np]]

    return {
        'mode': 'frozen_graph',
        'checkpoint_path': get_frozen_graph_path('faster_rcnn_resnet101_coco'),
        'input_tensors': ['image_tensor:0'],
        'output_tensors': ['detection_boxes:0', 'detection_scores:0',
                           'detection_classes:0', 'num_detections:0'],
        'post_processing_fn': post_process_fn,
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
def get_model_fn(model_name, batch_size=1):
    if model_name == 'mobilenet_v1_224':
        return mobilenet_v1_224(batch_size)
    elif model_name == 'ssd_mobilenet_v1_coco':
        return ssd_mobilenet_v1_coco(batch_size)
    elif model_name == 'faster_rcnn_resnet101_coco':
        return faster_rcnn_resnet101_coco(batch_size)
    else:
        raise Exception("Could not find network with name %s" % model_name)
