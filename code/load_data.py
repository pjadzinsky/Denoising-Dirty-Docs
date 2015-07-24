'''
Just load train and train_cleaned samples

Images can be of arbitrary size but for the time being I'm going to constrain them to 540x248 pixels

'''


from PIL import Image
import os
import numpy as np
import pdb

def load_data(sets_to_load):
    '''
    sets_to_load is a number in base 10 or a string representng the same number in binary notation ('0b111' is 7)
    where each digit in binary representation is a data to either load (1) or skip (0)
    bit 0:  train
    bit 1:  cleaned
    bit 2:  test
    bit 3:  thresh
    bit 4:  filtered

    load_data(3)    loads only train and cleaned
    load_data(7)    loads train, cleaned and test
    '''

    if type(sets_to_load)==int:
        sets_to_load = bin(sets_to_load)

    # invert sets_to_load such that the less significant bit is sets_to_load[0]
    sets_to_load = sets_to_load[:1:-1]      # inverts the string and removes part corresponding to '0b'

    #train=True, cleaned=True, test=True, thresh=True, filtered=True):
    folders = ['train', 'train_cleaned', 'test', 'train_thresh', 'train_filtered']

    output = []

    for bit, folder in zip(sets_to_load, folders):
        if bit=='1':
            files = os.listdir(folder)

            # each folder will load images into a different list. Each list starts empty
            output.append([])

            #pdb.set_trace()
            for f in files:
                try:
                    # try loading the image
                    im = Image.open(os.path.join(folder, f))
                    size = im.size
                    image = im.getdata()
                    #TODO for the time being, I'm limiting image to be 540x248
                    image = np.array(image)[:540*248].astype(np.float32)

                    # normalize images
                    image = (image - image.min())/(image.max() - image.min())

                    assert(image.max()==1)
                    assert(image.min()==0)
                    output[-1].append(image)
                except:
                    pass

            output[-1] = np.array(output[-1]).reshape(-1, 1, 248, 540)

    return tuple(output)
