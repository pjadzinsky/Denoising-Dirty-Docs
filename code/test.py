from keras.models import Sequential, Graph
from keras.layers.core import Layer, Dense, Activation, Merge, Reshape, Flatten, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSample2D
from keras.layers.extra import Pad_End
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, History, Callback#, SnapshotPrediction
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from code import load_data
from theano import tensor
import numpy as np
import pdb
from matplotlib import cm
import matplotlib.pyplot as plt
import h5py
import re
import os

class model(object):
    def __init__(self):
        pdb.set_trace()
        model = Sequential()

        model.add(Convolution2D(1, 2, 3, 3, border_mode='same', activation='relu'))

        model.add(MaxPooling2D(poolsize=(2,2)))

        model.add(UpSample2D(size=(2,2)))
        
        model.add(Pad_End(n=2, dim=1))

        model.add(Activation("softmax"))

        sgd = SGD(lr=.1)
        model.compile(sgd, 'mse')

        self.model = model

            
    def fit(self, X, Y, nb_epoch, save_models=[], logs={}, validation_split=0.1, X2=None):
        # X, Y are 4D arrays such that X.shape is (number of samples, color channels, height, width)
        # and X[0,:,:,:] is an image (shapes for Y are the same as for X)
        loss_file = os.path.join(self.model_path, self.loss_file)
        try:
            next_epoch = self.last_epoch+1
        except:
            next_epoch = 0

        if os.path.isfile(loss_file):
            overwrite = confirm_overwrite_file(loss_file)

            if overwrite=='n':
                return 'Aborting: do not overwrite'

        io_dict = {'input':X, 'output':Y}

        if X2 is not None:
            io_dict['input2'] = X2

        #savemodels = SaveModels()
        history = History()
        checkpointer = MyModelCheckpoint(self.model_path, self.model_name, next_epoch, save_models, verbose=1, save_best_only=False)
        #checkpred = SnapshotPrediction(filepath=model + '_prediction.hdf5')

        self.graph.fit(io_dict, nb_epoch=nb_epoch, batch_size=32, verbose=1,
                callbacks=[checkpointer, history], validation_split=validation_split)
        #self.graph.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpointer, checkpred],shuffle=False)
        
        try:
            self.loss = np.concatenate((self.loss, np.array(history.history['output'])), axis=0)
        except:
            self.loss = np.array(history.history['output'])

        if len(save_models):
            self.make_loss_file()

