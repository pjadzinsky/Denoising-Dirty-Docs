'''
Common functions that will be used by all models
'''
import matplotlib.pyplot as plt

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

