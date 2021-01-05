import numpy as np
from numpy.core.function_base import linspace
import pandas as pd
import random
import matplotlib.pyplot as plt

def signal(u):
    return int(abs(u)/u)

def EQM(amount, w, X, d):
    e = 0
    for i in range(len(X)):
        xk = np.array(X[i])
        u = np.matmul(w, xk.T)
        e = e + (d[i] - u) ** 2
        
    return e / amount

def plot(max_epoch, y):
    x = range(max_epoch)
    plt.plot(x,y)
    plt.show()

def train_adalaine_helper(n,epsilon, weights, data, result):
    epoch = 0
    w = np.array(weights)
    amount = len(data)
    Eant, Eatual = 0,EQM(amount, w, data, result)
    y = []
    while abs(Eatual - Eant) > epsilon:
        y.append(Eatual)
        Eant = Eatual
        for s in range(len(data)):
            xk = np.array(data[s])
            u = np.matmul(w, xk.T)
            tmp = n * (result[s] - u)
            val = np.multiply(tmp, xk)
            w = np.add(w, val)
        epoch = epoch + 1
        Eatual = EQM(amount, w, data, result)
    
    plot(epoch, y)
    
    return w, epoch

def test_perceptron(w,X):
    A, B = [], []
    for i in range(len(X)):
        u = np.matmul(w,X[i].T)
        y = signal(u)
        if y == 1:
            A.append(i+1)
        else: 
            B.append(i+1)
    return A,B

def get_set(filepath):
    return np.array(pd.read_csv(filepath,header=1))

def prepare_set(filepath):
    data = get_set(filepath)
    set = []
    d = []
    for i in data:
        set.append(i[:len(i)-1])
        d.append(i[-1])
    return np.array(set), np.array(d)

def training():
    n = 0.0025
    epsilon = 10**-6
    print(epsilon)
    print("N:",n)
    data_set, d = prepare_set('./apendice2.csv')
    all_weights = []
    for i in range(5):
        w = []
        for j in range(5):
            w.append(round(random.random(), 2))
        
        
        new_w, epoch = train_adalaine_helper(n,epsilon,w,data_set,d)
        print('Epoch:',epoch)
        print('Initial weights',w)
        print('New weights: ',new_w)
        print('\n')
        all_weights.append(new_w)
    return all_weights


def test_optimal(ww):
    amostras = np.array(get_set('./amostra2.csv'))

    for i in range(len(ww)):
        a,b = test_perceptron(ww[i], amostras)
        print("Teste #",i)
        print(a,'\n\n',b)


ww = training()
test_optimal(ww)
