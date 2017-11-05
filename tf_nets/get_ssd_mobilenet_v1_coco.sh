#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    cd tf_nets

    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
    tar -xvf ssd_mobilenet_v1_coco_11_06_2017.tar.gz

    rm ssd_mobilenet_v1_coco_11_06_2017.tar.gz
    mv ssd_mobilenet_v1_coco_11_06_2017 ssd_mobilenet_v1_coco

    cd $CWD
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi
