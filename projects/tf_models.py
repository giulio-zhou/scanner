import numpy as np
import pickle
import tensorflow as tf
from skimage import img_as_ubyte
from skimage.transform import resize

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
        mobilenet_v1(resized_inputs, num_classes=1001,
                     is_training=False, global_pool=True)

    def post_process_fn(input_columns, outputs):
        num_outputs = len(input_columns)
        serialize_fn = lambda x: np.ndarray.dumps(x.squeeze())
        return [[serialize_fn(outputs[0][i]) for i in range(num_outputs)]]

    return {
        'mode': 'python',
        'checkpoint_path': get_single_checkpoint_path(
            'mobilenet', 'mobilenet_v1_1.0_224.ckpt'),
        'input_tensors': ['image_tensor:0'],
        # 'output_tensors': ['MobilenetV1/Logits/AvgPool_1a/AvgPool:0'],
        'output_tensors': ['MobilenetV1/Logits/global_pool:0'],
        'post_processing_fn': post_process_fn,
        'session_feed_dict_fn': lambda sess, input_tensors, cols: \
            {input_tensors[0]: input_pre_process_fn(cols, batch_size)},
        'model_init_fn': create_mobilenet_model
    }

def draw_tf_bounding_boxes(
        image_np, boxes, scores, classes, num_detections, class_ids_to_names):
    import utils.visualization_utils as vis_util
    category_index = {
        i: {'name': name, 'id': i, 'display_name': name} \
          for i, name in class_ids_to_names.items()
    }
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.3)
    return image_np

def ssd_mobilenet_v1_coco_feature_extractor(batch_size=1):
    def post_process_fn(input_columns, outputs):
        num_outputs = len(input_columns)
        serialize_fn = lambda x: np.ndarray.dumps(x.squeeze().flatten())
        return [[serialize_fn(outputs[0][i]) for i in range(num_outputs)]]

    return {
        'mode': 'frozen_graph',
        'checkpoint_path': get_frozen_graph_path('ssd_mobilenet_v1_coco'),
        'input_tensors': ['image_tensor:0'],
        # 'output_tensors': ['FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6:0'],
        'output_tensors': ['FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/Relu6:0'],
        'post_processing_fn': post_process_fn,
        'session_feed_dict_fn': \
            lambda sess, input_tensors, cols: {input_tensors[0]: cols[0]}
    }

def ssd_mobilenet_v1_coco_detection_features(batch_size=1):
    def post_process_fn(input_columns, outputs):
        from constants import coco_class_ids_to_names
        boxes, scores, classes, num_detections = outputs
        scores, classes = np.squeeze(scores), np.squeeze(classes)
        # has person
        labels = np.zeros(4)
        for class_id, score in zip(classes, scores):
            class_name = coco_class_ids_to_names[class_id]
            if class_name == 'person':
                labels[0] = score >= 0.5
                labels[1] = score
                break
        # has car/truck/bus
        vehicle_classes = ["car", "truck", "bus"]
        for class_id, score in zip(classes, scores):
            class_name = coco_class_ids_to_names[class_id]
            if class_name in vehicle_classes:
                labels[2] = score >= 0.5
                labels[3] = score
                break
        return [[np.ndarray.dumps(labels)]]

    return {
        'mode': 'frozen_graph',
        'checkpoint_path': get_frozen_graph_path('ssd_mobilenet_v1_coco'),
        'header': ['has_person', 'person_conf', 'has_vehicle', 'vehicle_conf'],
        'input_tensors': ['image_tensor:0'],
        'output_tensors': ['detection_boxes:0', 'detection_scores:0',
                           'detection_classes:0', 'num_detections:0'],
        'post_processing_fn': post_process_fn,
        'session_feed_dict_fn': \
            lambda sess, input_tensors, cols: {input_tensors[0]: cols[0]}
    }

def ssd_mobilenet_v1_coco(batch_size=1):
    def post_process_fn(inputs, outputs):
        from constants import coco_class_ids_to_names
        image_np = inputs[0][0]
        boxes, scores, classes, num_detections = outputs
        image_np = draw_tf_bounding_boxes(
            image_np, boxes, scores, classes,
            num_detections, class_ids_to_names)
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
            lambda sess, input_tensors, cols: {input_tensors[0]: cols[0]}
    }

def faster_rcnn_resnet101_coco(batch_size=1):
    def post_process_fn(inputs, outputs):
        from constants import coco_class_ids_to_names
        image_np = inputs[0][0]
        boxes, scores, classes, num_detections = outputs
        image_np = draw_tf_bounding_boxes(
            image_np, boxes, scores, classes,
            num_detections, class_ids_to_names)
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
            lambda sess, input_tensors, cols: {input_tensors[0]: cols[0]}
    }

def yolo_v2_model(model_path, batch_size=1):
    def create_yolo_v2_model(K):
        score_threshold, iou_threshold = 0.3, 0.5
        from keras.models import load_model
        from constants import coco_classes as class_names
        from constants import yolo_anchors as anchors
        from yad2k.models.keras_yolo import yolo_eval, yolo_head, yolo_eval_batch
        yolo_model = load_model(model_path)
        anchors = np.array(anchors).reshape(-1, 2)
        yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
        input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes, frames = yolo_eval_batch(
            yolo_outputs,
            input_image_shape,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            batch_size=batch_size)
    return create_yolo_v2_model

def yolo_v2(batch_size=1):
    model_height, model_width = 416, 416
    model_path = 'tf_nets/yolo_v2/yolo.h5'
    input_imgs = tf.placeholder('uint8', [None, None, None, 3], name='imgs')
    resized_imgs = tf.image.resize_images(
        tf.cast(input_imgs, tf.float32), [model_height, model_width]) / 255.

    def pre_process_fn(input_columns, batch_size):
        batched_inputs = input_pre_process_fn(input_columns, batch_size)
        return batched_inputs

    def post_process_fn(inputs, outputs):
        from constants import coco_classes as class_names
        class_ids_to_names = \
            {i: class_names[i] for i in range(len(class_names))}
        output_imgs = []
        all_boxes, all_scores, all_classes, all_frames = outputs
        all_boxes /= model_height # Normalized coordinates.
        for i in range(len(inputs[0])):
            image_np = inputs[0][i]
            idx = np.where(all_frames == i)
            boxes, scores, classes = \
                all_boxes[idx], all_scores[idx], all_classes[idx]
            # Pad boxes, scores, classes if necessary
            if len(boxes) == 1:
                boxes = np.vstack([boxes, np.zeros((1, 4))])
                scores = np.array([scores[0], 0])
                classes = np.array([classes[0], 0])
            image_np = draw_tf_bounding_boxes(
                image_np, boxes, scores, classes,
                len(boxes), class_ids_to_names)
            image_np = img_as_ubyte(image_np)
            output_imgs.append(image_np)
        return [output_imgs]

    return {
        'mode': 'keras',
        'checkpoint_path': model_path,
        'input_tensors': \
            ['input_1:0', 'Placeholder_112:0', # input_image_shape
             'batch_normalization_1/keras_learning_phase:0'],
        'output_tensors': ['output_boxes:0', 'output_scores:0',
                           'output_classes:0', 'output_frames:0'],
        'post_processing_fn': post_process_fn,
        'session_feed_dict_fn': lambda sess, input_tensors, cols: \
            {input_tensors[0]: \
                 sess.run(resized_imgs, feed_dict={
                     input_imgs: pre_process_fn(cols, batch_size)}),
             input_tensors[1]: [model_height, model_width],
             input_tensors[2]: 0},
        'model_init_fn': yolo_v2_model(model_path, batch_size)
    }

def yolo_v2_detection_labels(batch_size=1):
    model_height, model_width = 416, 416
    model_path = 'tf_nets/yolo_v2/yolo.h5'
    input_imgs = tf.placeholder('uint8', [None, None, None, 3], name='imgs')
    resized_imgs = tf.image.resize_images(
        tf.cast(input_imgs, tf.float32), [model_height, model_width]) / 255.

    def pre_process_fn(input_columns, batch_size):
        batched_inputs = input_pre_process_fn(input_columns, batch_size)
        return batched_inputs

    def post_process_fn(inputs, outputs):
        from constants import coco_classes as class_names
        class_ids_to_names = \
            {i: class_names[i] for i in range(len(class_names))}
        output_annotations = []
        all_boxes, all_scores, all_classes, all_frames = outputs
        all_boxes /= model_height # Normalized coordinates.
        for i in range(len(inputs[0])):
            output_npy = []
            idx = np.where(all_frames == i)
            boxes, scores, classes = \
                all_boxes[idx], all_scores[idx], all_classes[idx]
            for j in range(len(boxes)):
                entry = \
                    [class_ids_to_names[classes[j]], scores[j]] + list(boxes[j])
                output_npy.append(entry)
            output_npy = np.array(output_npy)
            output_annotations.append(np.ndarray.dumps(output_npy))
        return [output_annotations]

    return {
        'mode': 'keras',
        'checkpoint_path': model_path,
        'header': ['object_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'],
        'input_tensors': \
            ['input_1:0', 'Placeholder_112:0', # input_image_shape
             'batch_normalization_1/keras_learning_phase:0'],
        'output_tensors': ['output_boxes:0', 'output_scores:0',
                           'output_classes:0', 'output_frames:0'],
        'post_processing_fn': post_process_fn,
        'session_feed_dict_fn': lambda sess, input_tensors, cols: \
            {input_tensors[0]: \
                 sess.run(resized_imgs, feed_dict={
                     input_imgs: pre_process_fn(cols, batch_size)}),
             input_tensors[1]: [model_height, model_width],
             input_tensors[2]: 0},
        'model_init_fn': yolo_v2_model(model_path, batch_size)
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
    elif model_name == 'ssd_mobilenet_v1_coco_detection_features':
        return ssd_mobilenet_v1_coco_detection_features(batch_size)
    elif model_name == 'ssd_mobilenet_v1_coco_feature_extractor':
        return ssd_mobilenet_v1_coco_feature_extractor(batch_size)
    elif model_name == 'faster_rcnn_resnet101_coco':
        return faster_rcnn_resnet101_coco(batch_size)
    elif model_name == 'yolo_v2':
        return yolo_v2(batch_size)
    elif model_name == 'yolo_v2_detection_labels':
        return yolo_v2_detection_labels(batch_size)
    else:
        raise Exception("Could not find network with name %s" % model_name)
