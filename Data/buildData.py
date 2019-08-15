import numpy as np
import pickle
from Data.MNIST import MNIST
# import Data.MNIST as MNIST

image_size = 28
no_of_different_labels = 10
image_pixels = image_size * image_size

data_path = ""
print("loading test data")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")
print("loaded test data")
print("loading training data")
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
print("loaded training data")

test_data[:10]

fac = 0.99 / 255
data = MNIST(np.asfarray(train_data[:, 1:]).T * fac + 0.01, np.asfarray(test_data[:, 1:]).T * fac + 0.01, np.asfarray(train_data[:, :1]).T, np.asfarray(test_data[:, :1]).T)

output = open('mnist.pkl', 'wb')
pickle.dump(data, output)
output.close()

this = "the end"
