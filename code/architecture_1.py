from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge
from keras.layers.convolutional import Convolution2D, CropImage
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback
from code.load_data import load_data
import numpy as np
import pdb

class model(object):
    def __init__(self, weights_file=None):
        nb_filters = [10,1]
        path1 = Sequential()
        path1.add(Convolution2D(nb_filters[0], 1, 3, 3, border_mode='same'))
        path1.add(Activation('relu'))
        
        path2 = Sequential()
        path2.add(Convolution2D(nb_filters[1], 1, 1, 1, border_mode='valid'))
        path2.add(Activation('relu'))

        model = Sequential()
        model.add(Merge([path1, path2], mode='concat', axis=1))
        model.add(Convolution2D(1, sum(nb_filters), 1, 1))
        model.add(Activation('sigmoid'))

        if weights_file is not None:
            model.load_weights(weights_file)

        sgd = SGD(lr=.15)
        model.compile(loss='mean_squared_error', optimizer=sgd)

        self.model = model

    def fit(self, X, Y, nb_epoch, logs={}):
        # X, Y are 3D arrays such that X.shape[0] is the number of samples
        # and X[0] is an image (shapes for Y are the same as for X)
        #pdb.set_trace()
        
        #savemodels = SaveModels()
        #history = LossHistory()
        checkpointer = MyModelCheckpoint(filepath='temp.hdf5', verbose=1, save_best_only=False)
        self.model.fit(X, Y, nb_epoch=nb_epoch, batch_size=32, show_accuracy=True, verbose=1, callbacks=[checkpointer])
        #self.model.fit(X, Y, nb_epoch=nb_epoch, batch_size=32, show_accuracy=True, verbose=1, callbacks=[checkpointer])

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
