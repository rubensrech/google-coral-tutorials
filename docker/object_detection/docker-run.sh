#!/bin/bash

DETECT_DIR=/mnt/4E0AEF320AEF15AD/RUBENS/det_out && mkdir -p $DETECT_DIR

docker run --name edgetpu-detect \
--rm --gpus all -it --privileged -p 6006:6006 \
--mount type=bind,src=${DETECT_DIR},dst=/tensorflow/models/research/learn_coco \
detect-tutorial-tf1