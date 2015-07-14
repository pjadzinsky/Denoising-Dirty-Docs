from code import utils, architecture_1, architecture_2, load_data
import code.histograms as h

train, train_cleaned, test = load_data.load_data()

h.plot_train_hist(train, train_cleaned)
h.obtain_freq(train, .75)
h.data_threshold(train, .2474)
hist_pred = h.predict(train, .749)

arch1 = architecture_1.model()
arch1.fit(train, train_cleaned)
arch1_pred = arch1.predict(train)
utils.display_prediction([train, train_cleaned, hist_pred, arch1_pred], 0, cols=2, labels=['ori', 'cleaned', 'hist', 'conv'])
