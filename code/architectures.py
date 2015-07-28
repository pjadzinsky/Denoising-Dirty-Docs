from keras.models import Sequential, Graph
from keras.layers.core import Layer, Dense, Activation, Merge, Reshape, Flatten, Permute
from keras.layers.convolutional import Convolution2D
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
    def __init__(self, model, f_sizes, nb_filters, model_name, weights_file=None):
        self.model_nb = model
        self.f_sizes = f_sizes
        self.nb_filters = nb_filters
        self.model_name = model_name
        self.model_path = 'model_weights'
        pdb.set_trace()
        self.model_regex = self.model_name.replace('{0}', '\d+')
        self.pred_name = self.model_name.replace('.hdf5', '_pred.hdf5')
        self.pred_path = 'predictions'
        self.pred_regex = self.model_regex.replace('.hdf5', '_pred.hdf5')

        if model_name.endswith('epoch{0}.hdf5'):
            pass
        else:
            raise ValueError('model_name should end with "epoch{0}.hdf5"')

        graph = Graph()

        graph.add_input(name='input', ndim=4)

        # Compute some nodes that will be reused over several layers
        graph.add_node(MeanImage(), 
                name='mean', input='input')
        
        graph.add_node(Permute((2,3,1)),
                name='input_permuted', input='input')

        graph.add_node(Permute((2,3,1)),
                name='mean_permuted', input='mean')

        if model==1:
            if type(f_sizes) == int:
                f_sizes = [f_sizes]
            
            if type(nb_filters) == int:
                nb_filters = [nb_filters]

            nb_pathways = max(len(f_sizes), len(nb_filters))
            
            if len(f_sizes) < nb_pathways:
                f_sizes *= nb_pathways
            
            if len(nb_filters) < nb_pathways:
                nb_filters *= nb_pathways
            
            if not len(f_sizes)==len(nb_filters):
                raise ValueError("Something is wrong. nb_filters has {0} and f_sizes has {1} points\
                        , they should be the same".format(len(nb_filters), len(f_sizes)))

            # Layer 1
            # -------
            # nb_pathways conv layers with different filter sizes that get concatenated with the
            # original image and a Mean intensity
            layers_to_concat = ['input_permuted', 'mean_permuted']

            model_name = ''
            for i in range(nb_pathways):
                graph.add_node(Convolution2D(nb_filters[i], 1, f_sizes[i], f_sizes[i], border_mode='same'),
                        name='scores_1_{0}'.format(i), input='input')

                graph.add_node(Permute((2,3,1)), 
                        name='scores_1_{0}_permuted'.format(i), input='scores_1_{0}'.format(i))
                
                layers_to_concat.append('scores_1_{0}_permuted'.format(i))

                model_name = model_name + '_{0}({1})'.format(f_sizes[i], nb_filters[i])

            model_name = 'model1' + model_name + 'epoch{0}.hdf5'

            graph.add_node(Activation('relu'),
                    name='activations_1_permuted', 
                    inputs=layers_to_concat,
                    merge_mode='concat')

            graph.add_node(Permute((3,1,2)),
                    name='activations_1', input='activations_1_permuted')

            # Layer 2
            # -------
            graph.add_node(Convolution2D(1, sum(nb_filters) + 2, 1, 1), 
                    name='scores_2', input='activations_1')

            graph.add_node(Activation('sigmoid'),
                    name='activations_2', input='scores_2')

            graph.add_output(name='output', input='activations_2')

        elif model==2:
            # Modification on model5, I'm adding another layer using conv2D at different scales

            # Compute some nodes that will be reused over several layers
            graph.add_node(MeanImage(), 
                    name='mean', input='input')
            
            graph.add_node(Permute((2,3,1)),
                    name='input_permuted', input='input')

            graph.add_node(Permute((2,3,1)),
                    name='mean_permuted', input='mean')

            # Layer 1
            # -------
            #, 3 conv layers with different filter sizes and a Mean intensity that get merged 
            graph.add_node(Convolution2D(nb_filters, 1, 5, 5, border_mode='same'),
                    name='scores_1a', input='input')

            graph.add_node(Convolution2D(nb_filters, 1, f_size, f_size, border_mode='same'),
                    name='scores_1b', input='input')

            graph.add_node(Permute((2,3,1)), 
                    name='scores_1a_permuted', input='scores_1a')

            graph.add_node(Permute((2,3,1)),
                    name='scores_1b_permuted', input='scores_1b')

            graph.add_node(Activation('relu'),
                    name='activations_1_permuted', 
                    inputs=['scores_1a_permuted', 'scores_1b_permuted', 'input_permuted', 'mean_permuted'],
                    merge_mode='concat')

            graph.add_node(Permute((3,1,2)),
                    name='activations_1', input='activations_1_permuted')

            # Layer 2
            # -------
            graph.add_node(Convolution2D(nb_filters, 2*nb_filters + 2, 3, 3, border_mode='same'), 
                    name='scores_2', input='activations_1')

            graph.add_node(Activation('relu'),
                    name='activations_2', input='scores_2')

            # Another instance of Layer 1 & 2

            # Layer 3 (identical to layer1)
            # -------
            graph.add_node(Convolution2D(nb_filters, nb_filters, 5, 5, border_mode='same'),
                    name='scores_3a', input='activations_2')

            graph.add_node(Convolution2D(nb_filters, nb_filters, f_size, f_size, border_mode='same'),
                    name='scores_3b', input='activations_2')

            graph.add_node(Permute((2,3,1)), 
                    name='scores_3a_permuted', input='scores_3a')

            graph.add_node(Permute((2,3,1)),
                    name='scores_3b_permuted', input='scores_3b')

            graph.add_node(Activation('relu'),
                    name='activations_3_permuted', 
                    inputs=['scores_3a_permuted', 'scores_3b_permuted', 'input_permuted', 'mean_permuted'],
                    merge_mode='concat')

            graph.add_node(Permute((3,1,2)),
                    name='activations_3', input='activations_3_permuted')

            # Layer 4 (identical to layer2)
            # -------
            graph.add_node(Convolution2D(nb_filters, 2*nb_filters + 2, 3, 3, border_mode='same'), 
                    name='scores_4', input='activations_3')

            graph.add_node(Activation('relu'),
                    name='activations_4', input='scores_4')

            # Layer 5
            # -------
            graph.add_node(Convolution2D(1, nb_filters, 1, 1),
                    name='scores_5', input='activations_4')

            graph.add_node(Activation('sigmoid'),
                    name='activations_5', input='scores_5')

            graph.add_output(name='output', input='activations_5')

        else:
            raise ValueError('Model {0} not recognized'.format(model))

        sgd = SGD(lr=.1)
        graph.compile(sgd, {'output':'mse'})

        self.graph = graph

    def model_init(self, weights_file=None):
        if weights_file is not None:
            self.graph.load_weights(weights_file)
            
            
    def fit(self, X, Y, nb_epoch, save_models=[], logs={}, validation_split=0.1):
        # X, Y are 4D arrays such that X.shape is (number of samples, color channels, height, width)
        # and X[0,:,:,:] is an image (shapes for Y are the same as for X)
        #pdb.set_trace()
        
        #savemodels = SaveModels()
        history = History()
        checkpointer = MyModelCheckpoint(self.model_path, self.model_name, 0, save_models, verbose=1, save_best_only=False)
        #checkpred = SnapshotPrediction(filepath=model + '_prediction.hdf5')

        #self.graph.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpointer, checkpred])
        self.graph.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1,
                callbacks=[checkpointer, history], validation_split=validation_split)
        #self.graph.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpointer, checkpred],shuffle=False)
        self.loss = np.array(history.history['output'])

        self.make_loss_file()

        return self

    def train(self, X, Y, nb_epoch, save_models=[]):
        # X, Y are 4D arrays such that X.shape is (number of samples, color channels, height, width)
        # and X[0,:,:,:] is an image (shapes for Y are the same as for X)
        #pdb.set_trace()
        
        datagen = ImageDataGenerator(
                width_shift_range=0.1
                )

        datagen.fit(X)
        model = self.graph

        #savemodels = SaveModels()
        #history = History()
        #checkpointer = MyModelCheckpoint(self.model_path, self.model_name, 0, save_models, verbose=1, save_best_only=False)

        history_loss = []
        for e in range(nb_epoch):
            print('Epoch', e)
            batch_loss = []
            # batch train with realtime data augmentation
            for X_batch, Y_batch in datagen.flow(X, Y):
                loss = model.train_on_batch({'input':X_batch, 'output':Y_batch})
                batch_loss.append(loss)


            history_loss.append(np.mean(batch_loss))
            print(history_loss[-1])

            if e in save_models:
                model.save_weights(self.model_name.format(e))

        # save loss history to file
        output = np.array(history_loss)
        f = h5py.File('model{0}_loss.hdf5'.format(self.model_nb), 'w')
        g = f.parent
        dset = g.create_dataset('output', output.shape, dtype=output.dtype)
        dset[:] = output
        f.flush()
        f.close()

    def save_predictions(self, data):
        '''
        Given model_regex to match saved model's weights as hdf5 files, load the weights for each hdf5 file
        and use them to predict 'data' saving the outputs to hdf5 input file + "_pred"
        '''
        model = self.graph
        pdb.set_trace()

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

    def make_loss_file(self):
        path = self.model_path
        loss_name = self.model_name.replace('epoch{0}.hdf5', 'loss.hdf5')
        loss_name = os.path.join(path, loss_name)
        if os.path.isfile(loss_name):
            import sys

            get_input = input
            if sys.version_info[:2] <= (2, 7):
                get_input = raw_input
            
            overwrite = get_input('[WARNING] %s already exists - overwrite? [y/n]' % (loss_name))
            while overwrite not in ['y', 'n']:
                overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
            if overwrite == 'n':
                return

        f = h5py.File(loss_name, 'w')
        g = f.parent
        g['loss'] = self.loss
        g.attrs['f_sizes'] = self.f_sizes
        g.attrs['nb_filters'] = self.nb_filters
        g.attrs['model_nb'] = self.model_nb

        f.flush()
        f.close()
        
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

def compare_losses(regex_str):
    '''
    load all hdf5 files that match regex_str and plot the 'loss' for each of them
    '''
    import re

    regex = re.compile(regex_str)

    fig, ax = plt.subplots(num='loss')

    loss_files = [f for f in os.listdir() if regex.search(f)]
    for f_name in loss_files:
        f = h5py.File(f_name, 'r')
        print(f_name)
        #pdb.set_trace()
        try:
            loss = np.array(f['loss'])
            label = f_name.replace('.hdf5', '').replace('_loss', '')
            ax.plot(loss, label=label)
        except:
            pass

        f.close()

    ax.legend()

def load_model(loss_file, weights_file):
    pdb.set_trace()
    f = h5py.File(loss_file, 'r')
    f_sizes = f.attrs['f_sizes']
    nb_filters = f.attrs['nb_filters']
    model_name = loss_file.replace('loss', 'epoch{0}')
    model_nb = int(f.attrs['model_nb'])

    def str2list(sList):
        return [int(i) for i in sList[1:-1].split(',')]

    if type(f_sizes) == np.ndarray:
        f_sizes = f_sizes.tolist()
    elif type(f_sizes) == np.string_:
        f_sizes = str2list(f_sizes)
    else:
        f_sizes = int(f_sizes)

    if type(nb_filters) == np.ndarray:
        nb_filters = nb_filters.tolist()
    elif type(nb_filters) == np.string_:
        nb_filters = str2list(nb_filters)
    else:
        nb_filters = int(nb_filters)


    f.close()
    mymodel = model(model_nb, f_sizes, nb_filters, model_name)
    
    try:
        mymodel.graph.load_weights(weights_file)
    except:
        pass

    return mymodel
    
