#!/bin/bash
module load cuda cudasamples
CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin
THEANO_FLAGS=device=gpu python code/train_script.py
