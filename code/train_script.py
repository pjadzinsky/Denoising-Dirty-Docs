import utils, architectures, load_data

train, train_cleaned, test = load_data.load_data()

model2 = architectures.model(3, nb_filters=3)
N = 10
model2.fit(train, train_cleaned, N, save_models=list(range(0,N,N/10)+[N-1]))

architectures.savePredictions(train[16,0,:,:])
