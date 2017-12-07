import caffe
import numpy as np
import tensorflow as tf
from skimage.transform import resize

def input_pre_process_fn(input_columns, batch_size):
    cols = input_columns[0]
    if len(cols) < batch_size:
        padding = [cols[0]] * (batch_size - len(cols))
        inputs = np.array(cols + padding)
    else:
        inputs = np.array(cols)
    return inputs

def create_feature_grid_tf(input_imgs, feature_tensors, factor):
    def make_feature_norm_map(height, width, activation_tensor,
                              min_norm=0, max_norm=2):
        norm = tf.norm(activation_tensor, axis=1)
        norm /= tf.sqrt(tf.cast(tf.shape(activation_tensor)[1], tf.float32))
        norm = tf.clip_by_value(norm, 0, max_norm)
        activations_mask = (norm - min_norm) / (max_norm - min_norm)
        activations_mask = tf.stack([activations_mask] * 3, axis=3)
        activations_mask = \
            tf.image.resize_images(activations_mask, [height, width])
        activations_mask = tf.cast(activations_mask * 255, tf.uint8)
        return activations_mask
    num_imgs = tf.shape(input_imgs)[0]
    img_height, img_width = tf.shape(input_imgs)[1], tf.shape(input_imgs)[2]
    feature_maps = []
    feature_height = img_height / factor
    feature_width = img_width / factor
    for i in range(factor * factor):
        if i < len(feature_tensors):
            feature_map = make_feature_norm_map(
                feature_height, feature_width, feature_tensors[i])
        else:
            # Pad the end of the grid with zeros.
            feature_map = \
                tf.zeros([num_imgs, feature_height, feature_width, 3], tf.uint8)
        feature_maps.append(feature_map)
    # Concatenate feature maps into final image.
    feature_map_rows = []
    for i in range(factor):
        print(feature_maps[i*factor:(i+1)*factor])
        row = tf.concat(feature_maps[i*factor:(i+1)*factor], 2)
        feature_map_rows.append(row)
    all_feature_maps = tf.concat(feature_map_rows, 1)
    imgs_and_feature_maps = tf.concat([input_imgs, all_feature_maps], 2)
    return imgs_and_feature_maps

def mobilenet(batch_size=1):
    input_imgs = tf.placeholder('uint8', [None, None, None, 3], name='imgs')
    resized_imgs = tf.image.resize_images(
        tf.cast(input_imgs, tf.float32), [224, 224])
    normalized_inputs = \
        0.017 * (resized_imgs - [123.68, 116.78, 103.94])

    def inference_fn(model, inputs):
        model.blobs['data'].data[...] = np.transpose(inputs, (0, 3, 1, 2))
        model.forward()
        # outputs = model.blobs['pool6'].data
        outputs = model.blobs['conv6/sep'].data
        outputs = np.squeeze(outputs)
        return [outputs]

    def post_process_fn(input_columns, outputs):
        num_outputs = len(input_columns)
        serialize_fn = lambda x: np.ndarray.dumps(x.squeeze())
        return [[serialize_fn(outputs[0][i]) for i in range(num_outputs)]]

    return {
        'model_prototxt_path': 'nets/mobilenet_deploy.prototxt',
        'model_weights_path': 'nets/mobilenet.caffemodel',
        'input_dims': [224, 224],
        'input_preprocess_fn': lambda sess, cols: sess.run(
            normalized_inputs, 
            feed_dict={input_imgs: input_pre_process_fn(cols, batch_size)}),
        'inference_fn': inference_fn,
        'post_processing_fn': post_process_fn,
    }

def mobilenet_feature_maps(batch_size=1):
    input_imgs = tf.placeholder('uint8', [None, None, None, 3], name='imgs')
    resized_imgs = tf.image.resize_images(
        tf.cast(input_imgs, tf.float32), [224, 224])
    normalized_inputs = \
        0.017 * (resized_imgs - [123.68, 116.78, 103.94])
    # Feature map-related tensors.
    layers = ["conv1", "conv2_1/sep", "conv2_2/sep", "conv3_1/sep",
              "conv3_2/sep", "conv4_1/sep", "conv4_2/sep", "conv5_1/sep",
              "conv5_2/sep", "conv5_3/sep", "conv5_4/sep", "conv5_5/sep",
              "conv5_6/sep", "conv6/sep"]
    feature_map_tensors = [tf.placeholder('float32', [None, None, None, None], 
                                          name=i) for i in layers]
    imgs_and_feature_maps = \
        create_feature_grid_tf(input_imgs, feature_map_tensors, 4)

    def inference_fn(model, inputs):
        outputs = []
        model.blobs['data'].data[...] = np.transpose(inputs, (0, 3, 1, 2))
        model.forward()
        for layer in layers:
            output_data = np.squeeze(model.blobs[layer].data)
            outputs.append(output_data)
        return [outputs]

    def post_process_fn(input_columns, outputs, tf_sess=None):
        num_outputs = len(input_columns)
        imgs = input_columns[0]
        input_dict = {input_imgs: imgs}
        for l, feature_npy in zip(layers, outputs[0]):
            input_dict[l + ':0'] = feature_npy[:num_outputs]
        imgs_with_feature_maps = \
            tf_sess.run(imgs_and_feature_maps, feed_dict=input_dict)
        output_imgs = [imgs_with_feature_maps[i] for i in range(num_outputs)]
        return [output_imgs]

    return {
        'model_prototxt_path': 'nets/mobilenet_deploy.prototxt',
        'model_weights_path': 'nets/mobilenet.caffemodel',
        'input_dims': [224, 224],
        'input_preprocess_fn': lambda sess, cols: sess.run(
            normalized_inputs, 
            feed_dict={input_imgs: input_pre_process_fn(cols, batch_size)}),
        'inference_fn': inference_fn,
        'post_processing_fn': post_process_fn,
    }

def get_model_fn(model_name, batch_size=1):
    if model_name == 'mobilenet':
        return mobilenet(batch_size)
    elif model_name == 'mobilenet_feature_maps':
        return mobilenet_feature_maps(batch_size)
    else:
        raise Exception("Could not find network with name %s" % model_name)
