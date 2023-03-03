#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
x = np.linspace(0,10)
y = 2*(x-3)**2 + x 

import matplotlib.pyplot as plt

plt.plot(x,y)
#%% Градиент

def optim(x0,a):  
    dydx = 6*(x0-3)+1
    dx = -a*dydx
    return dx

eps = 0.001
x0 = 8
k = 0
while True:
    X = x0 - optim(x0,0.01)
    k+=1
    print(X)
    if np.abs(X - x0)<eps:
        break
    x0=X
print(k)
print(X)
#%% Моментум
# Задание: реализовать градиентный спуск с помощью метода моментума
# В общую формулу градиентного спуска прибавить добавку - coef*dx. dx - это приращение на прошлом шаге
# coef принять равным 0.9