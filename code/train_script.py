import utils, architectures, load_data

train, train_cleaned, test = load_data.load_data()

model2 = architectures.model(2)
model2.fit(train, train_cleaned,2000)
