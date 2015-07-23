from keras.models import Sequential, Graph
from keras.layers.core import Layer, Dense, Activation, Merge, Reshape, Flatten, Permute
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, History, Callback#, SnapshotPrediction
import load_data
from theano import tensor
import numpy as np
import pdb
from matplotlib import cm
import matplotlib.pyplot as plt
import h5py


def model0():
    # conv layer + permutation
    graph = Graph()

    graph.add_input(name='input', ndim=4)

    graph.add_node(Convolution2D(1, 1, 5, 5, border_mode='same'),
            name='scores_1a', input='input')

    graph.add_node(Permute((2,3,1)), 
            name='permuted_1a', input='scores_1a')

    graph.add_node(Permute((3,1,2)),
            name='output_1', input='permuted_1a')

    graph.add_node(Activation('sigmoid'),
            name='activations_2', input='output_1')

    graph.add_output(name='output', input='activations_2')

    sgd = SGD(lr=.1)
    graph.compile(sgd, {'output':'mse'})


    graph.fit({'input':X, 'output':Y}, nb_epoch=2, batch_size=32, verbose=1)

def model1():
    graph = Graph()

    graph.add_input(name='input', ndim=4)

    graph.add_node(Convolution2D(1, 1, 5, 5, border_mode='same'),
            name='scores_1a', input='input')

    graph.add_node(Convolution2D(1, 1, 3, 3, border_mode='same'),
            name='scores_1b', input='input')

    graph.add_node(Permute((2,3,1)), 
            name='permuted_1a', input='scores_1a')

    graph.add_node(Permute((2,3,1)), 
            name='permuted_1b', input='scores_1b')

    graph.add_node(Activation('sigmoid'),
            name='permuted_activations_1', inputs=['permuted_1a', 'permuted_1b'], merge_mode='sum')

    graph.add_node(Permute((3,1,2)),
            name='layer_1_out', input='permuted_activations_1')

    graph.add_node(Convolution2D(1, 1, 1, 1),
            name='scores_2', input='layer_1_out')

    graph.add_node(Activation('sigmoid'),
            name='activations_2', input='scores_2')

    graph.add_output(name='output', input='activations_2')

    sgd = SGD(lr=.1)
    graph.compile(sgd, {'output':'mse'})


    graph.fit({'input':X, 'output':Y}, nb_epoch=2, batch_size=32, verbose=1)

X, Y, test = load_data.load_data()
print("Building and training model0, using convolve2D(... border_mode='same') and Permute")
model0()
print("model0 finished successfully")

print("Building and training model1, similar to model0 but using Activation(), inputs=['layerA', 'layerB'], merge_mode='concat'")

model1()
