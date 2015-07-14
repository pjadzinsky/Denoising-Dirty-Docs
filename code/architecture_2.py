from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD
from code.load_data import load_data
import numpy as np
import pdb

def model(X, Y, weights_file=None):
    # X, Y are 3D arrays such that X.shape[0] is the number of samples
    # and X[0] is an image (shapes for Y are the same as for X)
    model = Sequential()
    model.add(Convolution2D(1, 1, 5, 5, border_mode='same'))
    conv_layer = model.layers[0]
    conv_layer.get_output
    model.add(Activation('relu'))
    
    if weights_file is not None:
        model.load_weights(weights_file)

    model.compile(loss='mean_squared_error', optimizer='sgd')
    
    model.fit(X, Y, nb_epoch=5, batch_size=32, show_accuracy=True, verbose=1)

    return model

def predict(X, model):
    pdb.set_trace()
    prediction = model.predict(X)
    
    # for each image, compute a threshold
    temp = prediction.reshape(prediction.shape[0], -1)
    threshold = ( temp.max(axis=1) + temp.min(axis=1) )/2
    threshold = threshold.reshape(-1, 1, 1, 1)

    prediction = np.where(prediction < threshold, np.ones_like(prediction), np.zeros_like(prediction))
    return prediction
