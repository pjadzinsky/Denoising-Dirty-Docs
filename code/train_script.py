import utils, architectures, load_data

train, train_cleaned, test = load_data.load_data()

nb_filters = 1
epochs = 1000
save_every_epochs = 100
nb_model = 4
model2 = architectures.model(nb_model, nb_filters=nb_filters)
model2.fit(train, train_cleaned, epochs, save_models=list(range(0,epochs,save_every_epochs))+[epochs-1])

architectures.savePredictions(nb_model, nb_filters, train[16,0,:,:])
