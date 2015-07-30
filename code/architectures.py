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
    def __init__(self, model_nb, f_sizes, nb_filters, model_prefix, weights_file=None, compile_model=True):
        self.model_nb = model_nb
        self.f_sizes = f_sizes
        self.nb_filters = nb_filters
        self.model_prefix   = model_prefix
        self.model_path     = 'model_weights'
        self.pred_path      = 'predictions'
        self.model_name     = model_prefix + '_epoch{0}.hdf5'
        self.pred_name      = model_prefix + '_epoch{0}_pred.hdf5'
        self.model_regex    = model_prefix + '_epoch\d+.hdf5'
        self.pred_regex     = model_prefix + '_epoch\d+_pred.hdf5'
        self.loss_file      = model_prefix + '_loss.hdf5'

        if not compile_model:
            return 

        graph = Graph()

        graph.add_input(name='input', ndim=4)

        # Compute some nodes that will be reused over several layers
        graph.add_node(MeanImage(), 
                name='mean', input='input')
        
        graph.add_node(Permute((2,3,1)),
                name='input_permuted', input='input')

        graph.add_node(Permute((2,3,1)),
                name='mean_permuted', input='mean')
        
        layers_to_concat = ['input_permuted', 'mean_permuted']

        if model_nb in [3]:
            graph.add_input(name='input2', ndim=4)
            
            graph.add_node(Permute((2,3,1)),
                    name='input2_permuted', input='input2')

            layers_to_concat.append('input2_permuted')

        nb_extra_images = len(layers_to_concat)

        if model_nb==1:
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

        elif model_nb in [2,3]:
            '''
            in this model, each layer only has one convolutional pathway defined by f_size[i], nb_filters[i]
            At the end, the output of all conv pathways gets concat together along with original image, mean,
            threshold (potentialy more input images) and a single conv along channels is performed prior to 
            sigmoid.
            '''

            if nb_filters[0] != 1:
                raise ValueError(
                '''
                First point in nb_filters and f_sizes refers to the input image.
                nb_filters[0] = {0} and should be 1,
                f_sizes[0] is not used and can be any value
                '''.format(nb_filters[0])
                )

            if len(f_sizes) != len(nb_filters):
                raise ValueError('len(f_sizes)={0} and len(nb_filters)={1}, they should be the same'.format(
                    len(f_sizes), len(nb_filters)))

            # following layer only generates another layer called 'activations_0' that is identical to the input
            graph.add_node(Activation('linear'),
                name='activations_0',
                input='input')

            # loop through all filters and apply them at each stage.
            for L in range(1, len(f_sizes)):
                # Layer L
                # -------
                # conv layer with nb_filters[L], each of size f_sizes[L]
                
                graph.add_node(Convolution2D(nb_filters[L], nb_filters[L-1], f_sizes[L], f_sizes[L], border_mode='same'),
                        name='scores_{0}'.format(L), input='activations_{0}'.format(L-1))

                graph.add_node(Activation('relu'),
                        name='activations_{0}'.format(L), 
                        input='scores_{0}'.format(L))

                graph.add_node(Permute((2,3,1)),
                        name='activations_{0}_permuted'.format(L),
                        input='activations_{0}'.format(L))

                layers_to_concat.append('activations_{0}_permuted'.format(L))


            graph.add_node(Activation('linear'),
                    name='concatenation_permuted',
                    inputs=layers_to_concat)

            graph.add_node(Permute((3,1,2)),
                    name='concatenation',
                    input='concatenation_permuted')

            graph.add_node(Convolution2D(1, sum(nb_filters[1:]) + nb_extra_images, 1, 1),
                    name='final_conv',
                    input='concatenation')

            graph.add_node(Activation('sigmoid'),
                    name='sigmoid',
                    input='final_conv')

            graph.add_output(name='output', input='sigmoid')

        else:
            raise ValueError('Model {0} not recognized'.format(model_nb))

        sgd = SGD(lr=.1)
        graph.compile(sgd, {'output':'mse'})

        self.graph = graph

            
    def fit(self, X, Y, nb_epoch, save_models=[], logs={}, validation_split=0.1, X2=None):
        # X, Y are 4D arrays such that X.shape is (number of samples, color channels, height, width)
        # and X[0,:,:,:] is an image (shapes for Y are the same as for X)
        loss_file = os.path.join(self.model_path, self.loss_file)
        if os.path.isfile(loss_file):
            overwrite = confirm_overwrite_file(loss_file)

            if overwrite=='n':
                return 'Aborting: do not overwrite'

        io_dict = {'input':X, 'output':Y}

        if X2 is not None:
            io_dict['input2'] = X2

        #savemodels = SaveModels()
        history = History()
        checkpointer = MyModelCheckpoint(self.model_path, self.model_name, 0, save_models, verbose=1, save_best_only=False)
        #checkpred = SnapshotPrediction(filepath=model + '_prediction.hdf5')

        self.graph.fit(io_dict, nb_epoch=nb_epoch, batch_size=32, verbose=1,
                callbacks=[checkpointer, history], validation_split=validation_split)
        #self.graph.fit({'input':X, 'output':Y}, nb_epoch=nb_epoch, batch_size=32, verbose=1, callbacks=[checkpointer, checkpred],shuffle=False)
        self.loss = np.array(history.history['output'])

        if len(save_models):
            self.make_loss_file()


    def save_predictions(self, X, X2=None):
        '''
        Given model_regex to match saved model's weights as hdf5 files, load the weights for each hdf5 file
        and use them to predict 'X' saving the outputs to hdf5 input file + "_pred"
        '''
        model = self.graph

        io_dict = {'input':X}

        if X2 is not None:
            if type(X2)==np.ndarray:
                io_dict['input2'] = X2
            else:
                raise ValueError("can't recognize data type")

        regex = re.compile(self.model_regex)
        model_files = [f for f in os.listdir(self.model_path) if regex.search(f)]
        
        pdb.set_trace()
        if not os.path.isdir(self.pred_path):
            os.mkdir(self.pred_path)

        for f in model_files:
            model.load_weights(os.path.join(self.model_path, f))

            prediction = model.predict(io_dict)['output']

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
            fig_name = self.model_name.replace('epoch{0}.hdf5', 'im{0}.png').format(nb_img)

        regex = re.compile(self.pred_regex)
        pred_files = [f for f in os.listdir(pathin) if regex.search(f)]

        if not os.path.isdir(pathout):
            os.mkdir(pathout)

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
        ''' 
        write into a loss file the ndarray 'loss' and all necessary
        parameters to be able to re-initialize the model
        '''
        f = h5py.File(os.path.join(self.model_path, self.loss_file), 'w')
        g = f.parent
        g['loss'] = self.loss
        for k in ['model_nb', 'f_sizes', 'nb_filters', 'model_prefix']:
            try:
                g.attrs[k] = getattr(self, k)
            except:
                pass

        f.flush()
        f.close()
        
    def model_definition_str(self):
        '''
        Generate some type of nice string with the relevant model parameters
        '''
        text = 'M{0}'.format(self.model_nb)

        for size, nb in zip(self.f_sizes, self.nb_filters):
            text = text + ('_{0}({1})'.format(size, nb))

        return text

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

def compare_losses(regex_str, path='model_weights'):
    '''
    load all hdf5 files that match regex_str and plot the 'loss' for each of them
    '''
    import re

    regex = re.compile(regex_str)

    fig, ax = plt.subplots(num='loss')

    loss_files = [f for f in os.listdir(path) if regex.search(f)]
    print('comparing loss for files: {0}'.format(str(loss_files)))

    regex = re.compile('\d+')     # extract number from hdf5 file name
    for f_name in loss_files:
        loss_file = os.path.join(path, f_name)
        model = generate_model_from_loss_file(loss_file, compile_model=False)
        #f = h5py.File(os.path.join(path, f_name), 'r')
        try:
            #loss = np.array(f['loss'])
            #label = f_name.replace('.hdf5', '').replace('_loss', '')
            file_nb = regex.findall(f_name)[0]
            label = '#' + file_nb + model.model_definition_str()
            ax.plot(model.loss, label=label)
        except:
            pass

    ax.legend()

def generate_model_from_loss_file(loss_file, compile_model=True):
    # load from loss_file all needed parameters to recreate model initialization
    # model, f_sizes, nb_filters, model_name, weights_file=None):
    pdb.set_trace()
    f = h5py.File(loss_file, 'r')
    model_nb = int(f.attrs['model_nb'])
    f_sizes = f.attrs['f_sizes']
    nb_filters = f.attrs['nb_filters']
    model_prefix = f.attrs['model_prefix']
    loss = np.array(f['loss'])

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
    #if compile_model:

    pdb.set_trace()
    mymodel = model(model_nb, f_sizes, nb_filters, model_prefix,
            compile_model=compile_model)
    #else:
    #    mymodel['model_nb'] = model_nb
    #    mymodel['f_sizes = f_sizes
    #    mymodel.nb_filters = nb_filters
    #    mymodel.model_name = model_name
    #    mymodel.extra_images_path = extra_images_path
    
    mymodel.loss = loss
    return mymodel

def confirm_overwrite_file(file):
    import sys

    get_input = input
    if sys.version_info[:2] <= (2, 7):
        get_input = raw_input
    
    overwrite = get_input('[WARNING] %s already exists - overwrite? [y/n]' % (file))
    while overwrite not in ['y', 'n']:
        overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')

    return overwrite
