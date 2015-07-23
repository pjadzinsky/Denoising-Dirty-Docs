import utils, architectures, load_data

train, train_cleaned, test = load_data.load_data()

nb_filters = 1
nb_model = 4

architectures.savePredictions(nb_model, nb_filters, train[16,0,:,:])
