import math
import numpy as np
from numpy.lib.function_base import disp


class NeuralNetwork(object):
    def __init__(self):
        self.inputLayerSize = 3
        self.outputLayerSize = 2
        self.hiddenLayerSize = 4
        self.w1 = np.random.randn(3, 4)
        self.w2 = np.random.randn(4, 3)
        self.w3 = np.random.randn(3, 2)
        self.n = 0.015

    def foward(self, X):
        self.I1 = np.dot(self.w1, X.T)
        self.Y1 = np.insert(self.tanh(self.I1), 0, -1)
        self.I2 = np.dot(self.w2, self.Y1.T)
        self.Y2 = np.insert(self.tanh(self.I2), 0, -1)
        self.I3 = np.dot(self.w3, self.Y2.T)
        self.Y3 = np.insert(self.tanh(self.I3), 0, -1)
        return self.Y3

    def backward(self, d, X):
        y = self.foward(X)
        self.eqm = self.EQM(d, y)
        sig3 = np.multiply(-(d-y), self.tanhPrime(self.I3))
        delta3 = self.n * np.dot(self.Y2.T, sig3)
        self.w3 = self.w3 + delta3

    def EQM(self, d, y):
        e = 0
        for i in range(len(d)):
            e = e + (d[i] - y[i]) ** 2
        return e / len(d)

    def sum_of_diferences(self, d, y):
        e = 0
        for i in range(len(d)):
            e = (d[i] - y[i])
        return -e

    def tanh(self, z):
        return np.tanh(z)

    def tanhPrime(self, z):
        return 1.0 - np.tanh(z) ** 2


p = NeuralNetwork()
X = np.array([
    [-1, 0.2, 0.9, 0.4],
    [-1, 0.1, 0.3, 0.5],
    [-1, 0.9, 0.7, 0.8],
    [-1, 0.6, 0.4, 0.3]
])

d = np.array([
    [0.7, 0.3], [0.6, 0.4], [0.9, 0.5], [0.2, 0.8]
])
p.backward(d[0], X[0])
