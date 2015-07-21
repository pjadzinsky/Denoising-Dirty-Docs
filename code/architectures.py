from keras.models import Sequential, Graph
from keras.layers.core import Layer, Dense, Activation, Merge, Reshape, Flatten, Permute
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback, SnapshotPrediction
from code.load_data import load_data
from theano import tensor
import numpy as np
import pdb
from matplotlib import cm
import matplotlib.pyplot as plt
import h5py

class model(object):
    def __init__(self, model, weights_file=None, nb_filters=1):
        graph = Graph()

        graph.add_input(name='input', ndim=4)

        if model==1:
            # 3 conv layers with different filter sizes that get merged 
            graph.add_node(Convolution2D(nb_filters, 1, 1, 1, border_mode='valid'),
                    name='scores_1a', input='input')

            graph.add_node(Convolution2D(nb_filters, 1, 5, 5, border_mode='same'),
                    name='scores_1b', input='input')

            graph.add_node(Convolution2D(nb_filters, 1, 11, 11, border_mode='same'),
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

        if model==2:
            # Compute the average image across channels, it will be used as another input on each pixel latter on
            graph.add_node(MeanImage(), 
                    name='mean', input='input')

            # Concatenate input and mean image representations. 
            graph.add_node(Permute((2,3,1)), 
                    name='mean_permuted', input='mean')

            graph.add_node(Permute((2,3,1)),
                    name='input_permuted', input='input')

            graph.add_node(Activation('linear'),
                    name='concat_permuted', inputs=['mean_permuted', 'input_permuted'], merge_mode='concat')

            graph.add_node(Permute((3,1,2)),
                    name='concat', input='concat_permuted')
            
            # Finished concatenating images.

            graph.add_node(Convolution2D(1, 2, 1, 1, border_mode='same'),
                    name='scores_1', input='concat')

            graph.add_node(Activation('sigmoid'),
                    name='activations_1', input='scores_1')

            graph.add_output(name='output', input='activations_1')

        if weights_file is not None:
            graph.load_weights(weights_file)

        sgd = SGD(lr=.1)
        graph.compile('RMSprop', {'output':'mse'})

        self.graph = graph
        self.model_nb = model
            
            
    def fit(self, X, Y, nb_epoch, logs={}):
        # X, Y are 3D arrays such that X.shape[0] is the number of samples
        # and X[0] is an image (shapes for Y are the same as for X)
        #pdb.set_trace()
        
        #savemodels = SaveModels()
        #history = LossHistory()
        model = 'model{0}'.format(self.model_nb)
        checkpointer = MyModelCheckpoint(model, 0, [0,2,4], verbose=1, save_best_only=False)
        checkpred = SnapshotPrediction(filepath=model + '_prediction.hdf5')
        #self.graph.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpointer, checkpred])
        #self.model.fit(X, Y, nb_epoch=nb_epoch, batch_size=32, show_accuracy=True, verbose=1, callbacks=[checkpointer])
        self.graph.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpointer, checkpred],shuffle=False)

        return self

class MyModelCheckpoint(ModelCheckpoint):
    
    def __init__(self, prefix, epoch_offset, save_epochs, monitor='val_loss', verbose=0, save_best_only=False):
        super(MyModelCheckpoint, self).__init__('', monitor=monitor, verbose=verbose, save_best_only=save_best_only)
        self.prefix = prefix
        self.epoch_offset = epoch_offset
        self.save_epochs=save_epochs

    def on_epoch_end(self, epoch, logs={}):
        save_epochs = self.save_epochs
        pdb.set_trace()
        if epoch in save_epochs:
            self.filepath = '{0}_weights_epoch{1}.hdf5'.format(self.prefix, epoch)
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


class MeanImage(Layer):
    '''
        Generate an image (1, c, w, h) with mean intensity everywhere

        Dimensions of input are assumed to be (nb_samples, c, w, h).
        Return tensor has the same shape.
    '''
    def get_output(self, train=False):
        '''
        I want a 4D tensor with shape (n, c, r, w)
        Where out[j,i,:,:] are all the same value (the mean luminance for channel i, image j)

        I'll make the 4D tensor with 'tensordot' where the 1st tensor will have shape (n, c, 1, 1) (with the mean
        luminance value for each image and channel) and the 2nd tensor has shape (1, 1, r, w) and all the values are
        1
        '''
        X = self.get_input(train)           # X is a 4D tensor
        input_shape = X.shape


        X = X.mean(axis=2, keepdims=True)   # after this, X is a 4D tensor with shape (n, c, 1, w)
        X = X.mean(axis=3, keepdims=True)   # after this, X is a 4D tensor wiht shape (n, c, 1, 1)

        out = tensor.ones([1, 1, input_shape[2], input_shape[3]])     # shape is (1, 1, r, w) 

        #multiply X and out summing over (getting rid of) dimensions with 1s
        out = tensor.tensordot(X, out, [[2,3], [0,1]])
        return out

class SaveInput(Layer):
    '''
    Save input to hdf5 file
    Output of this layer is identical to input.
    '''
    def get_output(self, train=False):
        X = self.get_input(train)
        f = h5py.File('test', 'a')
        try:
            epoch = int(f['epochs']) + 1
        except:
            epoch = 0

        g = f.create_group(str(epoch))
        dset = g.create_dataset('input', X.shape, dtype=X.dtype)
        dset[:] = X
        
        f.flush()
        f.close()

        return X
