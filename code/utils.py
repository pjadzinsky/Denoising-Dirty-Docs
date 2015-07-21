'''
Common functions that will be used by all models
'''
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import h5py
import pdb

def display_prediction(data, index, labels=None, cols=None, rows=None):
    '''
    Display several versions of the same image.
    
    data:   iterable
            each member of the iterable is of the same shape as training data
    '''

    #pdb.set_trace()
    if cols is None:
        cols = 1
    if rows is None:
        rows = int(np.ceil(len(data)/cols))

    fig, axes = plt.subplots(rows, cols, num=2)

    for i, ax in enumerate(axes):
        ax.imshow(data[i][index,0,:,:])
        ax.axis('off')
        try:
            ax.set_title(labels[i])
        except:
            pass
    """
    for row in range(rows):
        for col in range(cols):
            i = row*cols+col
            ax[row, col].imshow(data[i][index,0,:,:])
            ax[row, col].axis('off')
            try:
                label = labels[i]
                ax[row, col].set_title(label)
            except:
                pass
    """

def save_pngs(snapshot_file, im_index, name_offset=0, epochs=None):
    f = h5py.File(snapshot_file, 'r')
    for i,k in enumerate(f):
        if epochs is not None:
            # only write images for epochs requested
            if i not in epochs:
                continue

        #pdb.set_trace()
        plt.imshow(f[k].get('output')[im_index,0,:,:], cmap=cm.Greys)
        plt.gcf().savefig('prediction_im{0}_epoch{1}.png'.format(im_index, name_offset + i))

    f.close()

