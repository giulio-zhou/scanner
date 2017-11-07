#!/bin/bash


CWD=`pwd`
DIR=`basename $CWD`

prog() {
    cd tf_nets

    wget -P mobilenet http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz
    cd mobilenet
    tar -xvf mobilenet_v1_1.0_224_2017_06_14.tar.gz
    rm mobilenet_v1_1.0_224_2017_06_14.tar.gz

    cd $CWD

    cp models/research/slim/nets/mobilenet_v1.py projects/
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi
