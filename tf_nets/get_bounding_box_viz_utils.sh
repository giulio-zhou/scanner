#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    git submodule update --init models

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
