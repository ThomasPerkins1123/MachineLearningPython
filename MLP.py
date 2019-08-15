import numpy as np
import math
from scipy.special import expit
import random


def sigmoid(a):
    return 1 / (1 + math.exp(-a))

class MLP:
    eta = 0
    E = 0
    epoch = 0
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
        hiddenLayer_W = []
        hiddenLayer_a = []
        hiddenLayer_o = []
        hiddenLayer_v = []
        hiddenLayer_G = []
        hiddenLayer_W.append(1 - np.random.rand(self.inputLayer_size + 1, self.hiddenLayer_size[0]) * 2)
        hiddenLayer_a.append([])
        hiddenLayer_o.append([])
        hiddenLayer_v.append([])
        hiddenLayer_G.append([])
        for i in range(1, self.hiddenLayer_N):
            hiddenLayer_W.append(1 - np.random.rand(self.hiddenLayer_size[i - 1] + 1, self.hiddenLayer_size[i]) * 2)
            hiddenLayer_a.append([])
            hiddenLayer_o.append([])
            hiddenLayer_v.append([])
            hiddenLayer_G.append([])
        self.outputLayer_W = 1 - np.random.rand(self.hiddenLayer_size[self.hiddenLayer_N - 1] + 1, self.outputLayer_size) * 2

        self.hiddenLayer_W = hiddenLayer_W
        self.hiddenLayer_a = hiddenLayer_a
        self.hiddenLayer_o = hiddenLayer_o
        self.hiddenLayer_v = hiddenLayer_v
        self.hiddenLayer_G = hiddenLayer_G

    def forwardPropogate(self, tImages):
        self.inputLayer_v = tImages
        self.inputLayer_v = np.append(self.inputLayer_v, np.ones((1, len(tImages[0]))), 0)
        self.hiddenLayer_a[0] = self.inputLayer_v.T.dot(self.hiddenLayer_W[0]).T
        self.hiddenLayer_o[0] = expit(self.hiddenLayer_a[0])
        self.hiddenLayer_v[0] = np.append(self.hiddenLayer_o[0], np.ones((1, len(tImages[0]))), 0)
        for i in range(1, self.hiddenLayer_N):
            self.hiddenLayer_a[i] = self.hiddenLayer_v[i-1].T.dot(self.hiddenLayer_W[i]).T
            self.hiddenLayer_o[i] = expit(self.hiddenLayer_a[i])
            self.hiddenLayer_v[i] = np.append(self.hiddenLayer_o[i], np.ones((1, len(tImages[0]))), 0)
        self.outputLayer_a = self.hiddenLayer_v[self.hiddenLayer_N - 1].T.dot(self.outputLayer_W).T
        self.outputLayer_o = expit(self.outputLayer_a)
        return self.outputLayer_o

    def costFunction(self, tImages, expected):
        self.forwardPropogate(tImages)
        temp = sum(np.square((expected - self.outputLayer_o.T).T))
        self.E = np.mean(temp)
        return self.E

    def gradDecent(self, expected):
        self.calcError(expected)
        dEdo = np.mean(self.outputLayer_e, 1)
        doda = np.mean(self.outputLayer_o * (1-self.outputLayer_o), 1)
        dadW = np.mean(self.hiddenLayer_v[self.hiddenLayer_N - 1], 1)
        derivative = np.atleast_2d(dEdo * doda).T
        self.outputLayer_G = (np.outer(derivative, dadW.T)).T
        if self.hiddenLayer_N == 1:
            print("to impliment later")
        else:
            dEdo = np.atleast_2d(np.mean(self.outputLayer_W.dot(derivative), 1)).T
            temp = np.atleast_2d(np.mean(self.hiddenLayer_o[self.hiddenLayer_N - 1], 1)).T
            doda = (temp * (1 - temp))
            derivative = dEdo[0:len(dEdo) - 1, :] * doda
            dadW = np.atleast_2d(np.mean(self.hiddenLayer_v[self.hiddenLayer_N - 2], 1)).T
            self.hiddenLayer_G[self.hiddenLayer_N - 1] = derivative.dot(dadW.T).T
            for i in range(self.hiddenLayer_N - 2, 0, -1):
                dEdo = np.atleast_2d(np.mean(self.hiddenLayer_W[i + 1].dot(derivative), 1)).T
                temp = np.atleast_2d(np.mean(self.hiddenLayer_o[i], 1)).T
                doda = (temp * (1 - temp))
                derivative = dEdo[0:len(dEdo) - 1, :] * doda
                dadW = np.atleast_2d(np.mean(self.hiddenLayer_v[i - 1], 1)).T
                self.hiddenLayer_G[i] = derivative.dot(dadW.T).T
            dEdo = np.atleast_2d(np.mean(self.hiddenLayer_W[1].dot(derivative), 1)).T
            temp = np.atleast_2d(np.mean(self.hiddenLayer_o[0], 1)).T
            doda = (temp * (1 - temp))
            derivative = dEdo[0:len(dEdo) - 1, :] * doda
            dadW = np.atleast_2d(np.mean(self.inputLayer_v, 1)).T
            self.hiddenLayer_G[0] = derivative.dot(dadW.T).T

    def alterWeights(self):
        self.outputLayer_W = self.outputLayer_W - self.eta * self.outputLayer_G
        for i in range(0, self.hiddenLayer_N - 1):
            self.hiddenLayer_W[i] = self.hiddenLayer_W[i] - self.eta * self.hiddenLayer_G[i]

    def backPropogate(self, tImages, expected):
        for i in range(0, len(tImages[0]) - 1):
            self.forwardPropogate(np.atleast_2d(tImages[:, i]).T)
            self.gradDecent(expected[i, :])
            self.alterWeights()
        self.epoch += 1

    def backPropogateBatch(self, tImages, expected):
        #doesnt Work
        self.forwardPropogate(tImages)
        self.gradDecent(expected)
        self.alterWeights()


    def calcError(self, expected):
        self.outputLayer_e = self.outputLayer_o - expected.T
        return self.outputLayer_e

    def getEpoch(self):
        return self.epoch



