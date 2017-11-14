#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    cd tf_nets

    wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
    tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz

    rm faster_rcnn_resnet101_coco_11_06_2017.tar.gz
    mv faster_rcnn_resnet101_coco_11_06_2017 faster_rcnn_resnet101_coco

    cd $CWD

    if [ ! -d projects/utils ]; then
        mkdir projects/utils
    fi
    if [ ! -f projects/utils/__init__.py ]; then
        touch projects/utils/__init__.py
    fi
    cp models/research/object_detection/utils/visualization_utils.py projects/utils
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi
