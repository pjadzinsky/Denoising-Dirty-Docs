import utils, architectures, load_data

train, cleaned= load_data.load_data(['train', 'train_cleaned'])
avg = load_data.load_data('train_avg')

epochs = 101
save_every_epochs = 10
nb_model = 3
f_size = [1, 3, 3, 3]
nb_filters = [1, 5, 5, 5]
name = 'm5'
model = architectures.model(nb_model, f_size, nb_filters, name)
model.fit(train, cleaned, epochs, save_models=list(range(0,epochs,save_every_epochs)), X2=avg)

