from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation, Merge, Reshape, Flatten, Permute
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback, SnapshotPrediction
from code.load_data import load_data
import numpy as np
import pdb
from matplotlib import cm
import matplotlib.pyplot as plt
import h5py

class model(object):
    def __init__(self, weights_file=None, nb_filters=1):
        graph = Graph()

        graph.add_input(name='input', ndim=4)

        graph.add_node(Convolution2D(nb_filters, 1, 1, 1, border_mode='valid'),
                name='scores_1a', input='input')

        graph.add_node(Convolution2D(nb_filters, 1, 3, 3, border_mode='same'),
                name='scores_1b', input='input')

        graph.add_node(Convolution2D(nb_filters, 1, 7, 7, border_mode='same'),
                name='scores_1c', input='input')

        graph.add_node(Permute((2,3,1)), 
                name='scores_1a_permuted', input='scores_1a')

        graph.add_node(Permute((2,3,1)),
                name='scores_1b_permuted', input='scores_1b')

        graph.add_node(Permute((2,3,1)),
                name='scores_1c_permuted', input='scores_1c')

        graph.add_node(Activation('relu'),
                name='activations_1_permuted', inputs=['scores_1a_permuted', 'scores_1b_permuted', 'scores_1c_permuted'])

        graph.add_node(Permute((3,1,2)),
                name='activations_1', input='activations_1_permuted')

        graph.add_node(Convolution2D(1, 3*nb_filters, 3, 3, border_mode='same'), 
                name='scores_2', input='activations_1')

        graph.add_node(Activation('sigmoid'),
                name='activations_2', input='scores_2')

        graph.add_output(name='output', input='activations_2')

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
        checkpointer = ModelCheckpoint(filepath='model_weights.hdf5', verbose=1, save_best_only=False)
        checkpred = SnapshotPrediction(filepath="model1_prediction.hdf5")
        #self.graph.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpointer, checkpred])
        #self.model.fit(X, Y, nb_epoch=nb_epoch, batch_size=32, show_accuracy=True, verbose=1, callbacks=[checkpointer])
        self.graph.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpointer, checkpred],shuffle=False)

class model2(object):
    def __init__(self, weights_file=None, nb_filters=1):
        model = Sequential()

        model.add(Convolution2D(nb_filters, 1, 3, 3, border_mode='same'))

        model.add(Activation('sigmoid'))

        if weights_file is not None:
            model.load_weights(weights_file)

        sgd = SGD(lr=.15)
        model.compile('RMSprop', 'mse')

        self.model = model

    def fit(self, X, Y, nb_epoch, logs={}):
        # X, Y are 3D arrays such that X.shape[0] is the number of samples
        # and X[0] is an image (shapes for Y are the same as for X)
        #pdb.set_trace()
        
        #savemodels = SaveModels()
        #history = LossHistory()
        checkpointer = MyModelCheckpoint(filepath='temp.hdf5', verbose=1, save_best_only=False)
        checkpred=SnapshotPrediction(filepath="model2_prediction.hdf5")
        #checkpred=SavePredictions(filepath="model2_ori.hdf5")
        #self.model.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpointer, checkpred])
        #self.model.fit(X, Y, nb_epoch=nb_epoch, batch_size=32, show_accuracy=True, verbose=1, callbacks=[checkpointer])
        self.model.fit(X, Y, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpred])

#class SaveModels(Callback):
class MyModelCheckpoint(ModelCheckpoint):
    
    def on_epoch_end(self, epoch, logs={}):
        self.filepath = 'weights_' + str(epoch) + '.hdf5'
        super(MyModelCheckpoint, self).on_epoch_end(epoch, logs)
        
class SavePredictions(Callback):
    def __init__(self, filepath, verbose=0):
        super(Callback, self).__init__()
        
        self.verbose = verbose
        self.filepath = filepath

    def on_epoch_begin(self, epoch, logs={}):
        # open hdf5 file to save predictions. On first epoch, if file is open it will be empty ('w').
        # On any other epoch file is opened mode 'a'
        import h5py
        
        filepath = self.filepath
        print('opening file, epoch=', epoch)
        try:
            self.epoch=epoch    # will be used as dictionary key when saving predictions
            if epoch==0:
                # Open HDF5 for saving
                import os.path

                # if file exists and should not be overwritten
                overwrite = False
                if not overwrite and os.path.isfile(filepath):
                    import sys
                    get_input = input
                    if sys.version_info[:2] <= (2, 7):
                        get_input = raw_input
                    overwrite = get_input('[WARNING] %s already exists - overwrite? [y/n]' % (filepath))
                    while overwrite not in ['y', 'n']:
                        overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
                    if overwrite == 'n':
                        return
                    print('[TIP] Next time specify overwrite=True in save_weights!')

                    self.f = h5py.File(filepath, 'w')
                else:
                    self.f = h5py.File(filepath, 'w')
            else:
                self.f = h5py.File(filepath, 'a')

            print('file open as ', self.f)
        except:
            self.f.close()

    def on_batch_begin(self, batch, logs={}):
        try:
            if batch==0:
                import pdb
                pdb.set_trace()
                X = self.model.batch
                if type(self.model)==Graph:
                    X = {name:value for (name, value) in zip(self.model.input_order, X)}
                elif type(self.model)==Sequential:
                    X = X[0]

                predictions = self.model.predict(X)

                # save to file
                f = self.f
                print('about to create group in file')
                g = f.create_group(str(self.epoch))
                print('group created')
                if type(self.model)==Graph:
                    for name in predictions:
                        dset = g.create_dataset(name, predictions[name].shape, dtype=predictions[name].dtype)
                        dset[:] = predictions[name]
                elif type(self.model)==Sequential:
                    # Check this works as expected
                    dset = g.create_dataset('output', predictions.shape, dtype=predictions.dtype)
                    dset[:] = predictions

                f.flush()
                f.close()
        except:
            self.f.close()

def save_pngs(snapshot_file, im_index, start_epoch=0):
    f = h5py.File(snapshot_file, 'r')
    for i,k in enumerate(f):
        #pdb.set_trace()
        plt.imshow(f[k].get('output')[im_index,0,:,:], cmap=cm.Greys)
        plt.gcf().savefig('prediction_im{0}_epoch{1}.png'.format(im_index, start_epoch + i))

