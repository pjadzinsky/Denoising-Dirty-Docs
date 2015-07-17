from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation, Merge, Reshape, Flatten
from keras.layers.convolutional import Convolution2D, CropImage
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback
from code.load_data import load_data
import numpy as np
import pdb

class model(object):
    def __init__(self, weights_file=None, flatten=False):
        # if flatten is set, training and test data has been flattened
        # reshape it into 248x540
        nb_filters = [1]
        graph = Graph()

        # independently of flatten, what comes out is 'input' as a 4D tensor
        if flatten:
            graph.add_input(name='flatten_input', ndim=2)
            graph.add_node(Reshape(1, 248, 540), name='input',
                    input='flatten_input')
        else:
            graph.add_input(name='input', ndim=4)

        graph.add_node(Convolution2D(nb_filters[0], 1, 3, 3, border_mode='same'),
                name='scores_1a', input='input')
        graph.add_node(Activation('sigmoid'), name='activations_1a', input='scores_1a')
        
        #graph.add_node(Convolution2D(nb_filters[1], 1, 1, 1, border_mode='valid'),
        #        name='scores_1b', input='input')
        #graph.add_node(Activation('relu'), name='activations_1b', input='scores_1b')

        graph.add_output(name='output', input='activations_1a')
        # Merge the two pathways
        #graph.add_node(Dense(
        #model = Sequential()
        #model.add(Merge([graph, graph], mode='concat', axis=1))
        #model.add(Convolution2D(1, sum(nb_filters), 1, 1))
        #model.add(Activation('sigmoid'))

        if flatten:
            graph.add_node(Flatten())

        if weights_file is not None:
            graph.load_weights(weights_file)

        sgd = SGD(lr=.15)
        graph.compile('RMSprop', {'output':'mse'})

        self.graph = graph

    def fit(self, X, Y, nb_epoch, logs={}):
        # X, Y are 3D arrays such that X.shape[0] is the number of samples
        # and X[0] is an image (shapes for Y are the same as for X)
        #pdb.set_trace()
        
        #savemodels = SaveModels()
        #history = LossHistory()
        checkpointer = MyModelCheckpoint(filepath='temp.hdf5', verbose=1, save_best_only=False)
        self.graph.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpointer])
        #self.model.fit(X, Y, nb_epoch=nb_epoch, batch_size=32, show_accuracy=True, verbose=1, callbacks=[checkpointer])
        predictions = self.graph.predict({'input':X})
        return predictions

    def predict(self, X, binary_flag=False):
        prediction = self.model.predict(X)
        
        if binary_flag:
            # for each image, compute a threshold
            temp = prediction.reshape(prediction.shape[0], -1)
            threshold = ( temp.max(axis=1) + temp.min(axis=1) )/2
            threshold = threshold.reshape(-1, 1, 1, 1)

            prediction = np.where(prediction > threshold, np.ones_like(prediction), np.zeros_like(prediction))

        prediction = 1 - prediction

        return prediction

#class SaveModels(Callback):
class MyModelCheckpoint(ModelCheckpoint):
    
    def on_epoch_end(self, epoch, logs={}):
        self.filepath = 'weights_' + str(epoch) + '.hdf5'
        super(MyModelCheckpoint, self).on_epoch_end(epoch, logs)

"""
class SavePredictions(Callback):
    def on_epoch_begin(self, epoch, logs-{}):
        X = 
        self.model.predict
"""
