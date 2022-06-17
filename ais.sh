#!/bin/bash

python -u eval_ais.py \
    --dataset_name static_mnist \
    --eval_sampler dula \
    --eval_step_size 0.1 \
    --sampling_steps 40 \
    --model resnet-64 \
    --buffer_size 10000 \
    --n_iters 300000 \
    --base_dist \
    --n_samples 500 \
    --eval_sampling_steps 300000 \
    --ema \
    --viz_every 1000 ;