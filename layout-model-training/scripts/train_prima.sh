#!/bin/bash

cd ../tools

python train_net.py \
    --dataset_name          test-layout \
    --json_annotation_train ../../home/GIT/Layoutparser-fast-api/examples/data/train.json \
    --image_path_train      ../../home/GIT/Layoutparser-fast-api/examples/data/images \
    --json_annotation_val   ../../home/GIT/Layoutparser-fast-api/examples/data/test.json \
    --image_path_val        ../../home/GIT/Layoutparser-fast-api/examples/data/images \
    --config-file           ../configs/prima/mask_rcnn_R_50_FPN_3x.yaml \
    OUTPUT_DIR  ../outputs/prima/mask_rcnn_R_50_FPN_3x/ \
    SOLVER.IMS_PER_BATCH 2