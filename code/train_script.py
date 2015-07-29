import utils, architectures, load_data

train, train_cleaned= load_data.load_data(3)

epochs = 101
save_every_epochs = 10
nb_model = 1
f_size = [3,5,21]
nb_filters = [5,1,1]
name = 'model2_epoch{0}.hdf5'
model = architectures.model(nb_model, f_size, nb_filters, name)
model.fit(train, train_cleaned, epochs, save_models=list(range(0,epochs,save_every_epochs)))

