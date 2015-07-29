import utils, architectures, load_data

train, train_cleaned= load_data.load_data(3)

epochs = 101
save_every_epochs = 10
nb_model = 2
f_size = [1, 3,3]
nb_filters = [1, 5,5]
name = 'model3_epoch{0}.hdf5'
model = architectures.model(nb_model, f_size, nb_filters, name)
model.fit(train, train_cleaned, epochs, save_models=list(range(0,epochs,save_every_epochs)))

