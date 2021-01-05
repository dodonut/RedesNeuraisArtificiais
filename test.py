import numpy as np


def linear_combination(W, X):
    I = []
    if len(W) > 0 and type(W[0]) != list:
        t = 0
        for i in range(len(W)):
            t = t + W[i]*X[i]
        return t
    for i in W:
        I.append(linear_combination(i, X))
    return I


# ww1 = [[0.2, 0.4, 0.5], [0.3, 0.6, 0.7], [0.4, 0.8, 0.3]]
# ww2 = [0.2, 0.6, 0.8]
# X = [1, 2, 3]
# print(linear_combination(ww2, X))

def act_line(I):
    y = []
    for i in I:
        y.append((1/math.cosh(i))**2)
    return y


def sigma(dd, yy, gl):
    sub = dd-yy
    return sub*gl


def backward(d, y, i):
    val = sigma(np.array(d), np.array(i), [1, 2, 3])
    return val


a = [1, 2]
b = [3, 4]
print(np.multiply(a, b))
