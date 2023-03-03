#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
x = np.linspace(-2,2)
y = 0.5*x + 3*np.sin(x) + np.random.randn(x.size)


#%%
class Poly():
    def __init__(self,x,y):
        #if x.size != y.size:
           # print('Размеры не совпадают')
           # raise NameError('dimention not equal')
        self.__verify_size(x, y)
        self.x = x
        self.y = y
        
    def __verify_size(self,x,y):
        if not isinstance(x, np.ndarray):
            raise NameError('Class must be ndarray')
            
        if not isinstance(y, np.ndarray):
            raise NameError('Class must be ndarray')
        
        if x.size != y.size:
            print('Размеры не совпадают')
            raise NameError('dimention not equal')
        
        
    def __len__(self):
        return len(self.x)
        
    
    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(x,y,'.')
        y_pred = 0
        for i in range(self.n+1):
            y_pred += self.coef[i]*x**i
        plt.plot(x,y_pred)
        
#%%
class PolyLSM(Poly):
    def __init__(self,x,y,n):
        super().__init__(x,y)
        self.n = n
    def get_poly(self):
        A = np.zeros([self.x.size,self.n+1])
        for i in range(self.n+1):
            A[:,i] = x**i
        coef = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y)
        t = np.linalg.inv(A.T.dot(A)) 
        print(np.linalg.cond(t))
        self.coef = coef
        
    

 
#%%
t = PolyLSM(x,y,30)
t.get_poly()

t.plot()
#print(t.__dict__)


