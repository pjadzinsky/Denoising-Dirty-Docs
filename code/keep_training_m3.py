import utils, architectures, load_data

train, cleaned= load_data.load_data(['train', 'train_cleaned'])
avg = load_data.load_data('train_avg')

epochs = 301
save_every_epochs = 10
weights_file = 'model_weights/m3_epoch100.hdf5'

model = architectures.generate_model_from_loss_file('model_weights/m3_loss.hdf5')
model.continue_fit(weights_file, train, cleaned, epochs, save_models=range(0,epochs, save_every_epochs), X2=avg)

