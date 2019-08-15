import numpy as np
import matplotlib.pyplot as plt
image_size = 28 # width and length
no_of_different_labels = 10
image_pixels = image_size * image_size
data_path = "Data/"
print("here 1")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")
print("here 2")
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
print("here 3")
test_data[:10]

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])


this = "the end"
