import caffe
import numpy as np
from skimage.transform import resize

def input_pre_process_fn(input_columns, batch_size):
    cols = input_columns[0]
    if len(cols) < batch_size:
        padding = [cols[0]] * (batch_size - len(cols))
        inputs = np.array(cols + padding)
    else:
        inputs = np.array(cols)
    return inputs

def mobilenet(batch_size=1):
    def preprocess_fn(input_columns):
        batched_inputs = input_pre_process_fn(input_columns, batch_size)
        resized_inputs = \
            np.array(map(lambda x: resize(x, [224, 224]), batched_inputs))
        normalized_inputs = \
            0.017 * (255. * resized_inputs - [123.68, 116.78, 103.94])
        return normalized_inputs

    def inference_fn(model, inputs):
        model.blobs['data'].data[...] = np.transpose(inputs, (0, 3, 1, 2))
        model.forward()
        outputs = model.blobs['pool6'].data
        outputs = np.squeeze(outputs)
        return [outputs]

    def post_process_fn(input_columns, outputs):
        num_outputs = len(input_columns)
        serialize_fn = lambda x: np.ndarray.dumps(x.squeeze())
        return [[serialize_fn(outputs[0][i]) for i in range(num_outputs)]]

    return {
        'model_prototxt_path': 'nets/mobilenet_deploy_caffe_group.prototxt',
        'model_weights_path': 'nets/mobilenet.caffemodel',
        'input_dims': [224, 224],
        'input_preprocess_fn': preprocess_fn,
        'inference_fn': inference_fn,
        'post_processing_fn': post_process_fn,
    }

def get_model_fn(model_name, batch_size=1):
    if model_name == 'mobilenet':
        return mobilenet(batch_size)
    else:
        raise Exception("Could not find network with name %s" % model_name)
