from keras.models import Sequential, Graph
from keras.layers.core import Layer, Dense, Activation, Merge, Reshape, Flatten, Permute
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, History, Callback#, SnapshotPrediction
from keras.regularizers import l2, activity_l2
from . import load_data
from theano import tensor
import numpy as np
import pdb
from matplotlib import cm
import matplotlib.pyplot as plt
import h5py
import re
import os

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

        elif model==2:
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

        elif model==3:
            # 3 conv layers with different filter sizes and a Mean intensity that get merged 
            graph.add_node(Convolution2D(nb_filters, 1, 5, 5, border_mode='same'),
                    name='scores_1a', input='input')

            graph.add_node(Convolution2D(nb_filters, 1, 11, 11, border_mode='same'),
                    name='scores_1b', input='input')

            # Compute the average image across channels, it will be used as another input on each pixel latter on
            graph.add_node(MeanImage(), 
                    name='mean', input='input')
            
            graph.add_node(Permute((2,3,1)), 
                    name='scores_1a_permuted', input='scores_1a')

            graph.add_node(Permute((2,3,1)),
                    name='scores_1b_permuted', input='scores_1b')

            graph.add_node(Permute((2,3,1)),
                    name='input_permuted', input='input')

            graph.add_node(Permute((2,3,1)),
                    name='mean_permuted', input='mean')

            graph.add_node(Activation('relu'),
                    name='activations_1_permuted', inputs=['scores_1a_permuted', 'scores_1b_permuted', 'input_permuted', 'mean_permuted'])

            graph.add_node(Permute((3,1,2)),
                    name='activations_1', input='activations_1_permuted')

            graph.add_node(Convolution2D(1, 2*nb_filters + 2, 5, 5, border_mode='same'), 
                    name='scores_2', input='activations_1')

            graph.add_node(Activation('sigmoid'),
                    name='activations_2', input='scores_2')

            graph.add_output(name='output', input='activations_2')

        elif model==4:
            # Extension of model 3.
            # Layer 1. Mixes filtered images with mean and input and passes through relu
            # Layer 2. Mixes layer1 with mean and input again and passes through relu
            # Layer 3. Output (sigmoid)
            #merge of filters, mean and input I'm adding another layer whereChanging layer that pulls together filters and mean to relu
            # and adding a sigmoid at the end.

            # Layer 1
            # =======
            #, 3 conv layers with different filter sizes and a Mean intensity that get merged 
            graph.add_node(Convolution2D(nb_filters, 1, 5, 5, border_mode='same'),
                    name='scores_1a', input='input')

            graph.add_node(Convolution2D(nb_filters, 1, 11, 11, border_mode='same'),
                    name='scores_1b', input='input')

            # Compute the average image across channels, it will be used as another input on each pixel latter on
            graph.add_node(MeanImage(), 
                    name='mean', input='input')
            
            graph.add_node(Permute((2,3,1)), 
                    name='scores_1a_permuted', input='scores_1a')

            graph.add_node(Permute((2,3,1)),
                    name='scores_1b_permuted', input='scores_1b')

            graph.add_node(Permute((2,3,1)),
                    name='input_permuted', input='input')

            graph.add_node(Permute((2,3,1)),
                    name='mean_permuted', input='mean')

            graph.add_node(Activation('relu'),
                    name='activations_1_permuted', inputs=['scores_1a_permuted', 'scores_1b_permuted', 'input_permuted', 'mean_permuted'], merge_mode='concat')

            graph.add_node(Permute((3,1,2)),
                    name='activations_1', input='activations_1_permuted')

            # Layer 2
            # =======
            graph.add_node(Convolution2D(1, 2*nb_filters + 2, 5, 5, border_mode='same'), 
                    name='scores_2', input='activations_1')

            graph.add_node(Activation('relu'),
                    name='activations_2', input='scores_2')

            # Layer 3
            # =======
            # mix this again with mean layer and input layer
            graph.add_node(Permute((2,3,1)),
                    name='activations_2_permuted', input='activations_2')

            graph.add_node(Activation('linear'),
                    name='activations_2_merged', inputs=['activations_2_permuted', 'mean_permuted', 'input_permuted'], merge_mode='concat')

            graph.add_node(Permute((3,1,2)),
                    name='merged_2', input='activations_2_merged')

            graph.add_node(Convolution2D(1, 3, 1, 1),
                    name='scores_3', input='merged_2')

            graph.add_node(Activation('sigmoid'),
                    name='activations_3', input='scores_3')

            graph.add_output(name='output', input='activations_3')

        elif model==5:
            # Modification on model4, I'm preprocessing data by normalizing between 0 and 1.

            # Layer 1
            # =======
            #, 3 conv layers with different filter sizes and a Mean intensity that get merged 
            graph.add_node(Convolution2D(nb_filters, 1, 5, 5, border_mode='same'),
                    name='scores_1a', input='input')

            graph.add_node(Convolution2D(nb_filters, 1, 11, 11, border_mode='same'),
                    name='scores_1b', input='input')

            # Compute the average image across channels, it will be used as another input on each pixel latter on
            graph.add_node(MeanImage(), 
                    name='mean', input='input')
            
            graph.add_node(Permute((2,3,1)), 
                    name='scores_1a_permuted', input='scores_1a')

            graph.add_node(Permute((2,3,1)),
                    name='scores_1b_permuted', input='scores_1b')

            graph.add_node(Permute((2,3,1)),
                    name='input_permuted', input='input')

            graph.add_node(Permute((2,3,1)),
                    name='mean_permuted', input='mean')

            graph.add_node(Activation('relu'),
                    name='activations_1_permuted', 
                    inputs=['scores_1a_permuted', 'scores_1b_permuted', 'input_permuted', 'mean_permuted'],
                    merge_mode='concat')

            graph.add_node(Permute((3,1,2)),
                    name='activations_1', input='activations_1_permuted')

            # Layer 2
            # =======
            graph.add_node(Convolution2D(nb_filters, 2*nb_filters + 2, 5, 5, border_mode='same'), 
                    name='scores_2', input='activations_1')

            graph.add_node(Activation('relu'),
                    name='activations_2', input='scores_2')

            # Layer 3
            # -------
            graph.add_node(Convolution2D(1, nb_filters, 1, 1),
                    name='scores_3', input='activations_2')

            graph.add_node(Activation('sigmoid'),
                    name='activations_3', input='scores_3')

            graph.add_output(name='output', input='activations_3')
            
        elif model==6:
            # Modification on model5, adding regularization.

            # Layer 1
            # =======
            #, 3 conv layers with different filter sizes and a Mean intensity that get merged 
            graph.add_node(Convolution2D(nb_filters, 1, 5, 5, border_mode='same', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)),
                    name='scores_1a', input='input')

            graph.add_node(Convolution2D(nb_filters, 1, 11, 11, border_mode='same', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)), 
                    name='scores_1b', input='input')

            # Compute the average image across channels, it will be used as another input on each pixel latter on
            graph.add_node(MeanImage(), 
                    name='mean', input='input')
            
            graph.add_node(Permute((2,3,1)), 
                    name='scores_1a_permuted', input='scores_1a')

            graph.add_node(Permute((2,3,1)),
                    name='scores_1b_permuted', input='scores_1b')

            graph.add_node(Permute((2,3,1)),
                    name='input_permuted', input='input')

            graph.add_node(Permute((2,3,1)),
                    name='mean_permuted', input='mean')

            graph.add_node(Activation('relu'),
                    name='activations_1_permuted', 
                    inputs=['scores_1a_permuted', 'scores_1b_permuted', 'input_permuted', 'mean_permuted'],
                    merge_mode='concat')

            graph.add_node(Permute((3,1,2)),
                    name='activations_1', input='activations_1_permuted')

            # Layer 2
            # =======
            graph.add_node(Convolution2D(nb_filters, 2*nb_filters + 2, 5, 5,
                border_mode='same',
                W_regularizer=l2(0.01),
                activity_regularizer=activity_l2(0.01)
                ), 
                name='scores_2', input='activations_1')

            graph.add_node(Activation('relu'),
                    name='activations_2', input='scores_2')

            # Layer 3
            # -------
            graph.add_node(Convolution2D(1, nb_filters, 1, 1, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)),
                    name='scores_3', input='activations_2')

            graph.add_node(Activation('sigmoid'),
                    name='activations_3', input='scores_3')

            graph.add_output(name='output', input='activations_3')
        else:
            raise ValueError('Model {0} not recognized'.format(model))

        sgd = SGD(lr=.1)
        graph.compile(sgd, {'output':'mse'})

        self.graph = graph
        self.model_nb = model
        self.model_name = 'model{0}'.format(model) + '_epoch{0}.hdf5'
        self.model_path = 'model_weights'
        self.model_regex = self.model_name.replace('{0}', '\d+')
        self.pred_name = self.model_name.replace('.hdf5', '_pred.hdf5')
        self.pred_path = 'predictions'
        self.pred_regex = self.model_regex.replace('.hdf5', '_pred.hdf5')

    def model_init(self, weights_file=None):
        if weights_file is not None:
            self.graph.load_weights(weights_file)
            
            
    def fit(self, X, Y, nb_epoch, save_models=[], logs={}):
        # X, Y are 3D arrays such that X.shape[0] is the number of samples
        # and X[0] is an image (shapes for Y are the same as for X)
        #pdb.set_trace()
        
        #savemodels = SaveModels()
        history = History()
        checkpointer = MyModelCheckpoint(self.model_path, self.model_name, 0, save_models, verbose=1, save_best_only=False)
        #checkpred = SnapshotPrediction(filepath=model + '_prediction.hdf5')

        #self.graph.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpointer, checkpred])
        self.graph.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpointer, history])
        #self.graph.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpointer, checkpred],shuffle=False)
        output = np.array(history.history['output'])
        f = h5py.File('model{0}_loss.hdf5'.format(self.model_nb), 'w')
        g = f.parent
        dset = g.create_dataset('output', output.shape, dtype=output.dtype)
        dset[:] = output
        f.flush()
        f.close()
        print(history.history)
        return self

    def save_predictions(self, data):
        '''
        Given model_regex to match saved model's weights as hdf5 files, load the weights for each hdf5 file
        and use them to predict 'data' saving the outputs to hdf5 input file + "_pred"
        '''
        model = self.graph

        regex = re.compile(self.model_regex)
        model_files = [f for f in os.listdir(self.model_path) if regex.search(f)]
        
        if not os.path.isdir(self.pred_path):
            os.mkdir(self.pred_path)

        for f in model_files:
            model.load_weights(os.path.join(self.model_path, f))

            prediction = model.predict({'input':data})['output']

            #plt.imshow(prediction['output'][0,0,:,:], cmap=cm.Greys_r)
            #plt.savefig('prediction_model{0}_epoch{1}.png'.format(nb_model, epoch))
            nameout = os.path.join(self.pred_path, f.replace('.hdf5', '_pred.hdf5'))

            fout = h5py.File(nameout, 'w')
            g = fout.parent
            dset = g.create_dataset('output', prediction.shape, prediction.dtype)
            dset[:] = prediction

            fout.flush()
            fout.close()

    def save_images(self, ori_set, nrows, ncols, nb_img, fig_name=None):
        '''
        open all hdf5 files that match self.pred_regex, extract nb_img from them and plot them with the 
        given number of rows and cols

        inputs:
        ------
            ori_set:    numpy.array
                        something like train or test, the 4D numpy array used for training

            nrows/cols: int

            nb_img:     int
                        which out of all predictions in the files to draw
                        plots f['output'][nb_img, 0, :, :]

            fig_name:   str
                        name on figure, if None will default to 'Im_#'

        '''
        pathin = self.pred_path
        pathout = self.pred_path
        if fig_name is None:
            fig_name = 'Im_{0}'.format(nb_img)

        regex = re.compile(self.pred_regex)
        pred_files = [f for f in os.listdir(pathin) if regex.search(f)]

        #pdb.set_trace()
        if not os.path.isdir(pathout):
            os.mkdir(pathout)

        #pdb.set_trace()
        fig, ax = plt.subplots(num='images_to_save', nrows=nrows, ncols=ncols)

        if ax.ndim==1:
            ax = ax.reshape(-1,1)

        # plot original image
        ax[0,0].imshow(ori_set[nb_img, 0, :, :], cmap=cm.Greys_r)
        ax[0,0].set_title('Original')
        ax[0,0].axis('off')

        for i,f in enumerate(pred_files):
            col = np.mod(i+1, ncols)
            row = (i+1)//ncols
            fid = h5py.File(os.path.join(pathin, f), 'r')
            im = fid['output']
            ax[row, col].imshow(im[nb_img, 0, :, :], cmap=cm.Greys_r)
            ax[row, col].axis('off')

            fid.close()

            #if we have more predictions than requested panels break
            if i+2 == nrows*ncols:      # +2 because: one +1 comes from the fact that ax[0,0] is the original image
                                        #           : another +1 comes from the fact that I want to know if I have an ax for next image
                break

        fig.savefig(os.path.join(pathout, fig_name))
        
class MyModelCheckpoint(ModelCheckpoint):
    '''
    Save models as it learns. Models are saved under self.path with name self.name after relapcing a literal "{0}"
    by the epoch number. An exammple of a valid self.name = 'model5_weights_{0}.hdf5'
    '''
    def __init__(self, path, name, epoch_offset, save_epochs, monitor='val_loss', verbose=0, save_best_only=False):
        super(MyModelCheckpoint, self).__init__('', monitor=monitor, verbose=verbose, save_best_only=save_best_only)
        self.path = path
        self.name = name
        self.epoch_offset = epoch_offset
        self.save_epochs=save_epochs

        if not os.path.isdir(path):
            os.mkdir(path)

    def on_epoch_end(self, epoch, logs={}):
        save_epochs = self.save_epochs
        if epoch in save_epochs:
            self.filepath = os.path.join(self.path, self.name.format(epoch))
            super(MyModelCheckpoint, self).on_epoch_end(epoch, logs)
        

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

