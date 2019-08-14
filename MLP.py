import numpy as np
import random

class MLP:
    eta = 0
    E = 0
    inputLayer_size = 0
    hiddenLayer_size = []
    outputLayer_size = 0
    hiddenLayer_W = []
    outputLayer_W = []
    hiddenLayer_N = 0
    inputLayer_v = []
    hiddenLayer_v = []
    hiddenLayer_a = []
    outputLayer_a = []
    hiddenLayer_o = []
    outputLayer_o = []
    outputLayer_e = []
    outputLayer_G = []
    hiddenLayer_G = []

    def __init__(self, inputLayer_size, hiddenLayer_size, outputLayer_size):
        self.eta = 0.2
        self.inputLayer_size = inputLayer_size
        self.hiddenLayer_size = hiddenLayer_size
        self.outputLayer_size = outputLayer_size
        self.hiddenLayer_N = len(hiddenLayer_size)
        # self.hiddenLayer_W.append(np.ones((self.inputLayer_size + 1, self.hiddenLayer_size[0])) * random.random())
        self.hiddenLayer_W.append(1 - np.random.rand(self.inputLayer_size + 1, self.hiddenLayer_size[0]) * 2)
        for i in range(1, self.hiddenLayer_N):
            self.hiddenLayer_W.append(1 - np.random.rand(self.hiddenLayer_size[i - 1] + 1, self.hiddenLayer_size[i]) * 2)
        self.outputLayer_W = 1 - np.random.rand(self.hiddenLayer_size[self.hiddenLayer_N - 1] + 1, self.outputLayer_size) * 2


        this = "the end"