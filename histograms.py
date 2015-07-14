'''
This is about the simples algorithm I can think of.
From the cleaned samples, compute the frequency of 1s (and 0s) (I'll
have to binarize train_cleaned for this) and then choose a single
threshold per image such that the frequency of 1s and 0s is mainteined
'''
import numpy as np
import matplotlib.pyplot as plt
import pdb

def obtain_freq(Y,threshold):
    '''
    Binarize Y and return the frequency of 0s

    I'm returning freq of 0s such that percentile(X, freq_of_0) will
    threshold the data at the right place

    inputs:

    Y:      cleaned data

    threshold:      pixel value. Anything above threshold in Y is taken as a '1'
                    anything below threshold is taken as a '0'
    '''
    binarized = np.where(Y>threshold, np.ones_like(Y), np.zeros_like(Y))
    result = 1-binarized.mean()

    print('When the pixel threshold is {0}, {1:3.2f}/{2:3.2f} of the data is below/above threshold'.format(threshold,
        result, 1-result))
    return result 

def data_threshold(train, freq_0):
    '''
    obtain the pixel value from data to maintein a zero frequency matching 'freq_0'
    '''
    if freq_0 < 1:
        print('Multiplying freq_0 by 100')
        freq_0*=100

    result = np.percentile(train, freq_0)

    print('To obtain {0:.2f} fraction of the data as "0" we need a threshold equal to {1:.2f}'.format(
        freq_0, result))
    return result


def predict(X, threshold):

    return np.where(X<threshold, X, np.ones_like(X))


def display_prediction(train, train_cleaned, prediction, index):
    '''
    Display one of the original images next to the prediction
    '''

    fig, ax = plt.subplots(3,1, num=2)
    ax[0].imshow(train[index,0,:,:])
    ax[0].axis('off')
    ax[0].set_title('Original')
    ax[1].imshow(train_cleaned[index,0,:,:])
    ax[1].axis('off')
    ax[1].set_title('Cleaned')
    ax[2].imshow(prediction[index,0,:,:])
    ax[2].axis('off')
    ax[2].set_title('Predicted')

def plot_train_hist(train, train_cleaned, bins=np.arange(0, 1.025, .05)):
    # compute a histogram of pixel intensities
    hist1, _ = np.histogram(train, normed=True, bins=bins)
   
    # compute a histogram of pixel intensities after normalizing each image between 0 and 1
    temp = train.reshape(train.shape[0], -1)
    temp_min = temp.min(axis=1).reshape(-1, 1, 1, 1)
    temp_max = temp.max(axis=1).reshape(-1, 1, 1, 1)
    normalized = (train - temp_min)/(temp_max - temp_min)
    hist2, _ = np.histogram(normalized, normed=True, bins=bins)
    
    # compute a histogram of cleaned data
    hist3, _ = np.histogram(train_cleaned, normed=True, bins=bins)

    #pdb.set_trace()
    fig, ax = plt.subplots(num=1)
    ax.plot(bins[:-1], hist1, label='raw')
    ax.plot(bins[:-1], hist2, label='normed')
    ax.plot(bins[:-1], hist3, label='cleaned')
    ax.legend(loc='upper left')
    plt.axis('image')

