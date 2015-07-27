import utils, architectures, load_data

train, train_cleaned= load_data.load_data(3)

epochs = 101
save_every_epochs = 10
nb_model = 8
f_size = [5,11,15]
nb_filters = [10,10,10]
model = architectures.model(nb_model, f_size, nb_filters)
model.fit(train, train_cleaned, epochs, save_models=list(range(0,epochs,save_every_epochs)))

