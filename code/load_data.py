'''
Just load train and train_cleaned samples

Images can be of arbitrary size but for the time being I'm going to constrain them to 540x248 pixels

'''


from PIL import Image
import os
import numpy as np
import pdb

def load_data():
    folders = ['train', 'train_cleaned', 'test']

    output = []

    for folder in folders:
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
                image = list(image)[:540*248]
                output[-1].append(image)
            except:
                pass

    train = np.array(output[0]).reshape(-1, 1, 248, 540)
    train_cleaned = np.array(output[1]).reshape(-1, 1, 248, 540)
    test = np.array(output[2]).reshape(-1, 1, 248, 540)

    train = train.astype(np.float32)
    train_cleaned = train_cleaned.astype(np.float32)
    test = test.astype(np.float32)

    train /= 255.0
    train_cleaned /= 255.0
    test /= 255.0

    return train, train_cleaned, test
