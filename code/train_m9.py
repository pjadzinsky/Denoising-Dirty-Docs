import utils, architectures, load_data

train, cleaned= load_data.load_data(['train', 'train_cleaned'])
avg = load_data.load_data('train_avg')

epochs = 301
save_every_epochs = 10
nb_model = 3
f_size = [1, 7, 7]
nb_filters = [1, 10, 10]
name = 'm9'

model = architectures.model(nb_model, f_size, nb_filters, name)
model.fit(train, cleaned, epochs, save_models=list(range(0,epochs,save_every_epochs)), X2=avg)

