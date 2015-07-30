'''
Just load train and train_cleaned samples

Images can be of arbitrary size but for the time being I'm going to constrain them to 540x248 pixels

'''


from PIL import Image
import os
import numpy as np
import pdb

def load_data(folder_list, max_images=None):
    '''
    folder_list:    list of str
                    each item is a relative or full path to a folder with images
    '''

    if type(folder_list) == str:
        folder_list = [folder_list]

    output = []

    for folder in folder_list:
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

            if len(output[-1])==max_images:
                break

        output[-1] = np.array(output[-1]).reshape(-1, 1, 248, 540)

    if len(output)==1:
        return output[0]
    else:
        return tuple(output)
