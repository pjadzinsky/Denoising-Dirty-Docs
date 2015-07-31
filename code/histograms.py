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
from scipy.signal import convolve2d
from scipy import misc
from os import listdir, mkdir
from os import path
import PIL.Image
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

def get_char_distance(image, plot_flag=False, verbose=0, dir='v'):
    # Compute the average distance in pixels between characters
    # Correlate the images with themself and find peak of FFT

    h, w = image.shape

    if plot_flag:
        fig, ax = plt.subplots(num='Correlations', nrows=2)

    if dir=='h':
        i=0
    elif dir=='v':
        i=1
    else:
        raise ValueError("dir not recognized, has to be either 'v' or 'h'")

    # i=0   horizontal correlation
    # i=1   vertical correlation

    line = image.mean(axis=i)
    line -= line.mean()

    # I want to do a circular correlation. I'm going to paste all but one point of 
    # line to itself making its new length 2*len(line)-1
    longline = np.concatenate((line[:-1], line))

    corr = np.correlate(longline, line)     # default mode is valid, corr will be of length len(line)

    # compute the FFT 
    fft = rfft(corr)
    freq = rfftfreq(len(corr))

    freq_of_max = freq[fft.argmax()]
    filter_size = np.round(1/freq_of_max)

    if verbose:
        print('max achieved at freq: {0}'.format(freq_of_max))
        print('Distance between symbols is {0} pixels'.format(filter_size))

    if plot_flag:
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

    if plot_flag:
        fig.savefig('Correlations.png')

    return filter_size

def filter_image(image, filter_size, threshold, plot_flag=False):
    '''
    Smooth image with a filter (2d convolution) of 'filter_size'
    Then threshold image such that 'threshold' fraction of pixels is 1 and 1-'threshold' pixels are 0
    '''
    filter= np.ones(filter_size) / (filter_size[0]*filter_size[1])
    filtered = convolve2d(image, filter, mode='same', fillvalue=image.mean())

    if threshold < 1:
        threshold *= 100

    thresh = filtered > np.percentile(filtered, threshold)

    if plot_flag:
        fig = plt.figure('filtered_image')
        fig.clf()
        fig, ax = plt.subplots(num='filtered_image', nrows=3)

        ax[0].imshow(image)
        ax[1].imshow(filtered)
        ax[2].imshow(thresh)

        fig.savefig('filtered image')
    
    return filtered, thresh
"""
def process_folder(path_str, filtered_str, thresh_str, threshold = 0.7):
    '''
    for each image in the folder, compute its filter_size and use it to filter the image
    
    output:
        filtered_str:       a folder that will be created if it doesn't exist
                            will hold all filtered images, each image with the same name
                            as the original one

        thresh_str:         idem filtered_str but for thresholded images.
        
    '''
    pdb.set_trace()
    if not path.isdir(filtered_str):
        mkdir(filtered_str)

    if not path.isdir(thresh_str):
        mkdir(thresh_str)

    files = listdir(path_str)
    
    for f in files:
        pil_object = PIL.Image.open(path.join(path_str,f))
        im = np.array(pil_object.getdata()).reshape(pil_object.size[::-1])
        plt.imshow(im)
        filter_size = get_char_distance(im)/2
        filter_size = np.where(filter_size > 50, [5,5], filter_size)
        
        filtered, thresh = filter_image(im, filter_size, threshold, plot_flag=True)
        
        misc.imsave(path.join(filtered_str, f), filtered)
        misc.imsave(path.join(thresh_str, f), thresh)
"""
def threshold_lines(im):
    mean_im = im.mean(axis=1, keepdims=True)

    mean_im -= mean_im.mean()

    final = np.tensordot(mean_im, np.ones((1, im.shape[1])), (1,0))
    return final

def process_folder(input_path, output_path, func, *vargs, **kargs):
    '''
    for each image in input_path, do out_im = func(im, **kwords) and save out_im in output_path
    '''
    pdb.set_trace()
    if not path.isdir(output_path):
        mkdir(output_path)

    if not path.isdir(input_path):
        raise ValueError(
                """
                input_path = {0} is not a valid folder
                """.format(input_path))

    files = listdir(input_path)
    
    for f in files:
        pil_object = PIL.Image.open(path.join(input_path,f))
        im = np.array(pil_object.getdata()).reshape(pil_object.size[::-1])
        
        out_im = func(im, *vargs, **kargs)
        
        misc.imsave(path.join(output_path, f), out_im)
