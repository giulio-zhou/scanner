#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    cd nets

    wget https://www.dropbox.com/s/7s50k69qcblguob/yolo_small.tar?dl=0
    tar -xvf yolo_small.tar?dl=0
    rm yolo_small.tar?dl=0

    cd $CWD
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi
