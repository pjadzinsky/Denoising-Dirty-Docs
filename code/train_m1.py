import utils, architectures, load_data

train, cleaned= load_data.load_data(['train', 'train_cleaned'])

epochs = 101
save_every_epochs = 10
nb_model = 1
f_size = [3, 7, 15]
nb_filters = 10
name = 'm1'
model = architectures.model(nb_model, f_size, nb_filters, name)
model.fit(train, cleaned, epochs, save_models=list(range(0,epochs,save_every_epochs)))

