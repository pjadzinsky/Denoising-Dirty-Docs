import utils, architectures, load_data

train, train_cleaned, test = load_data.load_data()

nb_filters = 3
#model2 = architectures.model(3, nb_filters=nb_filters)
N = 3
#model2.fit(train, train_cleaned, N, save_models=list(range(0,N,N/10)+[N-1]))
#model2.fit(train, train_cleaned, N, save_models=[0,1])

architectures.savePredictions(3, nb_filters, train[16,0,:,:])
