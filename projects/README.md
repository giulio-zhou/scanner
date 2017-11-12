# Feature vector generation and visualization

This directory contains application and helper code for running networks
in Scanner. One of the main parts of the code is a wrapper operator capable
of running Tensorflow models. The purpose of each file is listed as follows:

* `constants.py` - defines commonly used constants such as list of COCO classes
    and matching ids
* `create_resized_imgs.py` - optionally applies a network operator across a
    video segment (mostly for annotation purposes) and generates a stacked
    numpy array of resized images
* `generate_hashes.py` - given a `.toml` file as input, runs a Caffe model and
    creates a numpy array of feature vectors
* `make_obj_detect_box_features.py` - given a Tensorflow network name, extracts
    bounding box related features (expects rows of a numpy array as output, i.e.
    something like a CSV in the shape of a numpy array with header followed by
    entries) but in principle, this could be any features that the user deems
    relevant
* `make_obj_detect_box_video.py` - given a Tensorflow network name, draws
    bounding boxes into a video and outputs an mp4
* `run_tf_hashes.py` - given a Tensorflow network name, extracts feature vectors
    (similar to `generate_hashes.py` except for Tensorflow)
* `tf_models.py` - defines all Tensorflow models (Caffe models defined using
    the `.toml` format) by providing (inexhaustive list) input/output tensors,
    pre/post-processing functions, path to saved model parameters

Although these operations could technically be done all in one Scanner
pipeline, sometimes there are issues (e.g. Caffe has trouble resizing images at
the same time as running a network) and the isolation's preferable for now.

As also described above, the types of outputs that can be created from the above
pipelines include:

* `data.npy` - a numpy array of downsampled images intended to be viewed in a
    Tensorboard visualization
* `feature_vectors.npy` - a numpy array of feature vectors
* `labels.npy` - essentially a CSV-like table stored in a numpy array of dtype
    np.object

A typical feature generation run will create a `labels.npy` array that contains
just the frame numbers (no header for 1-column labels for Tensorboard). Running
`make_obj_detect_box_features.py` will replace `labels.npy`, so should be done
after generating feature vectors if desired.

## Setup
This setup assumes that you have a GPU on your machine. There's probably no
good reason to use the CPU version unless you're running Intel-Caffe on a
machine with a number of good CPUs.

NOTE: These are the directions that I follow after I've started running Scanner
on Docker. Follow the main Scanner directions to get Scanner running on Docker,
which will involve installing Docker as well as getting `nvidia-docker` and
`nvidia-docker-compose` installed if you would like to use a GPU.

First, pull the contents of this branch then recompile using
```
cd build
make -j\`nproc\`
```

If using anything from the Tensorflow models repo (visualization or Python TF
network definitions), first run
```
git submodule update --init models
```

Download pip Tensorflow by running
```
pip install tensorflow-gpu
```
Get cudnn 6.0 for cuda version 8.0 and put the `libcudnn.so.6` file into
`/usr/local/cuda/lib64` and run
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```
or append whichever directory you put the `.so` file.

## Applying networks using Scanner
Invoking the following
```
python projects/create_resized_imgs.py /var/home/data/jackson-town-square.mp4 output_dir ssd_mobilenet_v1_coco
python projects/run_tf_networks.py /var/home/data/jackson-town-square.mp4 ssd_mobilenet_v1_coco output_dir
python projects/make_obj_detect_box_features.py /var/home/data/jackson-town-square.mp4 ssd_mobilenet_v1_coco output_dir
```
will create downsampled images with applied bounding boxes, extract features,
and create a label array that has vehicle and person info.

## Visualization in Tensorboard
Before viewing the results in Tensorboard, first generate a `LOG_DIR` containing
tensors and metadata prepared in the Tensorboard format.

From the projects directory, run
```
python viz/embedding.py output_dir/feature_vectors.npy LOG_DIR output_dir/labels.npy output_dir/data.npy
```

Then run
```
tensorboard --logdir LOG_DIR --host <host_ip> --port <port_no>
```
If running remotely, forward the ssh using
```
ssh -L <local_port>:<remote_ip>:<remote_port> username@instance -N
```
or using the following on GCP
```
gcloud compute ssh username@instance --ssh-flag="-L" --ssh-flag="<local_port>:<remote_ip>:<remote_port>" --ssh-flag="-N"
```
and go to `localhost:<local_port>` in your browser.

## Writing a new (Tensorflow) model
All Tensorflow model getter logic are in the `tf_nets` directory.

There are three model types that are currently supported:
    `frozen`, `python`, and `keras`

Frozen models generally freeze the batch size, but permit you to use a binary
proto that contains both the network structure and weights (i.e. one file that
contains everything). Generally all that needs to be done here is to add a
script that downloads the frozen graph and then specify a path to the newly
downloaded graph.

Python models allow batching but are more complicated to set up, in that they'll
require that you somehow import the Python definition of a Tensorflow model. The
`Mobilenet_v1-224` model is a good example of this. The getter script in
`tf_nets` copies a `mobilenet.py` file into `projects/` where `tf_models.py` can
import it. One main reason for this is that many of the stock TF graphs contain
unremoved references to `SSTableReader`, which is an internal Google data reader
that doesn't exist in open-source Tensorflow.

Keras models require that `pipeline_instances_per_node` be set to 1, since Keras
has an issue with multiple Tensorflow graphs being populated simultaneously. The
behavior is otherwise similar to the Python model, other than requiring that the
backend variable `K` be passed into the `model_init_fn`.

To write a new model, just follow the examples of one of the models listed in
`tf_models.py`.
