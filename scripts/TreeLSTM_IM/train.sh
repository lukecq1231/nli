#!/bin/bash

# GPU
export THEANO_FLAGS='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer_including=cudnn,warn_float64=warn,lib.cnmem=0.95'

# CPU
# export THEANO_FLAGS='mode=FAST_RUN,device=cpu,floatX=float32'

python -u ./train.py > log.txt 2>&1 &




