#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import пакетов
import torch
import torch.nn as nn


#%%
# Создаем случайный тензор (картинку)
t = torch.rand(1,3,6,6) # B-batch x C-chanel x H-hight x W-width
s = nn.Conv2d(3, 2, 3) # Свертка с 2-я ядрами размером 3х3

out = s(t)
print(out)

r = nn.ReLU() # ReLU = max(input,0)
out = r(out)
print(out)

mp = nn.MaxPool2d(2,stride=2) # ядро 2х2, шаг по сетке = 2
out = mp(out)
print(out)

out = out.flatten() # преобразование из n-мерного тензора к вектору
print(out)

f = nn.Linear(8,2) # матричная операция
out = f(out)
print(out)

sm = nn.Softmax(dim=0) # переход от численных значений к вероятностям

out = sm(out)
print(out)

#%%
# Некоторые слои удобно собирать в блоки
ss = nn.Sequential(nn.Conv2d(3, 2, 3),nn.ReLU(),
                  nn.MaxPool2d(2,stride=2) )

t = torch.rand(1,3,6,6)
print(ss(t))






