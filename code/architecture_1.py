from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD
from code.load_data import load_data
import numpy as np
import pdb

class model(object):
    def __init__(self, weights_file=None):
        model = Sequential()
        model.add(Convolution2D(1, 1, 1, 1))
        #model.add(
        model.add(Activation('sigmoid'))
        
        if weights_file is not None:
            model.load_weights(weights_file)

        model.compile(loss='mean_squared_error', optimizer='sgd')

        self.model = model

    def fit(self, X, Y):
        # X, Y are 3D arrays such that X.shape[0] is the number of samples
        # and X[0] is an image (shapes for Y are the same as for X)
        
        self.model.fit(X, Y, nb_epoch=5, batch_size=32, show_accuracy=True, verbose=1)

    def predict(self, X):
        prediction = self.model.predict(X)
        
        # for each image, compute a threshold
        temp = prediction.reshape(prediction.shape[0], -1)
        threshold = ( temp.max(axis=1) + temp.min(axis=1) )/2
        threshold = threshold.reshape(-1, 1, 1, 1)

        prediction = np.where(prediction > threshold, np.ones_like(prediction), np.zeros_like(prediction))
        return prediction
