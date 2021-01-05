import numpy as np
import pandas as pd
import random

# samples = [
#     [-1,0.1,0.4,0.7],
#     [-1,0.3,0.7,0.2],
#     [-1,0.6,0.9,0.8],
#     [-1,0.5,0.7,0.1]
# ]

# d = np.array([1,-1,-1,1])

def signal(u):
    return int(abs(u)/u)

def train_perceptron_helper(n, weights, data, result):
    epoch = 0
    err = True
    w = np.array(weights)
    while err and epoch <= 1000:
        err = False
        for s in range(len(data)):
            xk = np.array(data[s])
            u = np.matmul(w, xk.T)
            y = signal(u)
            if y != result[s]:
                val = np.multiply(n * (result[s] - y), xk)
                w = np.add(w, val)
                err = True
        epoch = epoch + 1
    
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
    n = round(random.uniform(0,0.3),2)
    print("N:",n)
    data_set, d = prepare_set('./apendice1.csv')
    for i in range(5):
        w = []
        for j in range(4):
            w.append(round(random.random(), 2))
        
        
        new_w, epoch = train_perceptron_helper(n,w,data_set,d)
        print('Epoch:',epoch)
        print('Initial weights',w)
        print('New weights: ',new_w)
        print('\n')


def test_optimal():
    ww = np.array([
    [-50.71,25.63,40.71,-12.05],
    [-50.28,25.01,40.23,-11.91],
    [-50.77,25.63,40.60,-12.03],
    [-50.74,25.53,40.47,-12.02],
    [-50.27,25.50,40.60,-11.97]
    ])

    amostras = np.array(get_set('./amostra1.csv'))

    for i in range(len(ww)):
        a,b = test_perceptron(ww[i], amostras)
        print("Teste #",i)
        print(a,'\n\n',b)

# running()