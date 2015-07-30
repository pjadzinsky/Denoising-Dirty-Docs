from nose.tools import assert_equal, assert_true, assert_raises, assert_almost_equal
from code import architectures, load_data
from numpy.testing import assert_array_almost_equal
import pdb

train   = load_data.load_data('train', max_images=2)
cleaned = load_data.load_data('train_cleaned', max_images=2)
avg     = load_data.load_data('train_avg', max_images=2)

def setup():
  print("SETUP!")

def teardown():
  print("TEAR DOWN!")

def test_basic():
  print("I RAN!")

def model1():
    model = architectures.model(1, [3], [2], 't0')
    model.fit(train, cleaned, 1)

def model2():
    model = architectures.model(2, [1,3], [3, 3], 't1')
    model.fit(train, cleaned, 1)

def model3():
    model = architectures.model(3, [1,3], [3, 3], 't1')
    model.fit(train, cleaned, 1, X2=avg)


