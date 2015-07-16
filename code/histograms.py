'''
This is about the simples algorithm I can think of.
From the cleaned samples, compute the frequency of 1s (and 0s) (I'll
have to binarize train_cleaned for this) and then choose a single
threshold per image such that the frequency of 1s and 0s is mainteined
'''
import numpy as np
import matplotlib.pyplot as plt
import code.utils
from scipy.fftpack import rfft, rfftfreq
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


def plot_train_hist(train, train_cleaned, bins=np.arange(0, 1.025, .05)):
    # compute a histogram of pixel intensities
    #pdb.set_trace()
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
    ax.cla()
    ax.plot(bins[:-1], hist1, label='raw', linewidth=3)
    #ax.plot(bins[:-1], hist2, label='normed')
    ax.plot(bins[:-1], hist3, label='cleaned', linewidth=3)
    ax.legend(loc='upper left')
    ax.set_ylabel('Probability')
    ax.set_xlabel('Normalized Luminance')
    ax.annotate('white', xy=(bins[-2],hist3[-1]), xytext=(0.8,15), fontsize=18, arrowprops=dict(facecolor='black', shrink=0.05),)
    ax.annotate('black', xy=(bins[0],hist3[0]), xytext=(0.2,4), fontsize=18, arrowprops=dict(facecolor='black', shrink=0.05),)

    fig.savefig('Distributions.png', transparent=True)

def train_clean_diff(train, train_cleaned, bins=np.arange(0, 1.025, .05)):
    temp = train - train_cleaned
    hist, _ = np.histogram(temp, normed=True, bins=bins)
    fig, ax = plt.subplots(num='Noise')
    ax.cla()    # clears the axis
    ax.plot(bins[:-1], hist, lw=3)
    ax.set_xlabel('Luminance difference (ori - cleaned)')
    ax.set_ylabel('Probability')
    fig.savefig('Noise distribution.png', transparent=True)

def get_correlation(cleaned, image):
    # get and display the correlation of images with themself.

    nb_images, _, h, w = cleaned.shape

    """
    if axis = 0:
        im_set = [cleaned[i,0,:,:] for i in range(nb_images)]
        images = np.hstack(im_set).mean(axis=1)
    else:
        images = cleaned.reshape(h*nb_images, w).mean(axis=0)
    """
    fig, ax = plt.subplots(num='Correlations', nrows=2)

    #pdb.set_trace()
    for i in range(2):
        # i=0   vertical correlation
        # i=1   horizontal correlation

        line = cleaned[image,0,:,:].mean(i)
        line -= line.mean()

        # I want to do a circular correlation. I'm going to paste all but one point of 
        # line to itself making its new length 2*len(line)-1
        longline = np.concatenate((line[:-1], line))

        corr = np.correlate(longline, line)     # default mode is valid, will be of length len(line)

        # compute the FFT 
        fft = rfft(corr)
        freq = rfftfreq(len(corr))

        if i==0:
            label = 'vertical'
        else:
            label = 'horizontal'
        ax[0].plot(corr, lw=2, label = label)
        ax[0].set_xlabel('Pixels')
        ax[0].legend(loc='upper center')
    
        ax[1].plot(freq, fft, lw=2, label= label)
        ax[1].set_xlabel('Frequency')
        ax[1].legend(loc='upper center')

    fig.savefig('Correlations.png')
