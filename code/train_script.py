import utils, architectures, load_data

train, train_cleaned= load_data.load_data(3)

nb_filters = 10
epochs = 101
save_every_epochs = 10
nb_model = 6
model2 = architectures.model(nb_model, nb_filters=nb_filters)
model2.fit(train, train_cleaned, epochs, save_models=list(range(0,epochs,save_every_epochs)))

