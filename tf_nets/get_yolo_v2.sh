#!/bin/bash


CWD=`pwd`
DIR=`basename $CWD`

prog() {
    cd tf_nets

    mkdir yolo_v2

    cd YAD2K
    wget http://pjreddie.com/media/files/yolo.weights
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
    python yad2k.py yolo.cfg yolo.weights model_data/yolo.h5
    rm yolo.weights yolo.cfg
    mv model_data/yolo.h5 ../yolo_v2

    cd $CWD

    cp -r tf_nets/YAD2K/yad2k projects/
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi
