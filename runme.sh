#!/bin/bash

# Name the job in Grid Engine
#$ -N runme

#tell Grid Engine to use current directory
#$ -cwd

# Tell Grid Engine to notify job owner if job 'b'egins, 'e'nds, 's'uspended, is 'a'borted or 'n'o mail
#$ -m besan

# Tell Grid Engine to join normal output and error output into one file
#$ -j y

module load cuda cudasamples
CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin
THEANO_FLAGS=device=cpu python code/train_script.py
