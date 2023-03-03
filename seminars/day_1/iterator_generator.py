#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%% итератор
a = iter([1,2,3,4,5])
print(next(a))
print(next(a))


#%% генератор
A = [9,3,2,1]
## enumetate полезно использовать, когда работаем с итерируемыми объектами,
#  т.к. она выводит не только значение самого объекта но и его индекс
for item in enumerate(A):
    print(item[0],item[1])
    
for idx,val in enumerate(A):
    print(idx,val)

#
# Один из способов задания генератора
b = (c for c in range(5))
next(b)
next(b)
# это список
# b = [c for c in range(5)]

# создание генератора с помощью yield
def gen():
    for i in range(5):
        yield i
        
t = gen()
print(t)
next(t)


