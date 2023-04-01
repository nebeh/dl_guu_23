#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 15:27:54 2023

@author: strike
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
p = "DigitDataset/"
dir = os.listdir(p)
df = pd.DataFrame(columns =['path','label'])

path_to_data = []
class_name = []
for el in dir:
    if os.path.isdir(p+el):
        DIR = os.listdir(p+el)
        for label in DIR:
            path_to_data.append(p+el+"/"+label)
            class_name.append(el)
            
s = set(class_name)
df = pd.DataFrame(data={'path':path_to_data, 'label':class_name})

sh = df.shape[0]
array_a = np.random.permutation(range(sh))
id_train = round(0.8*sh)
id_test = round(0.1*sh)
id_val = round(0.1*sh)

train_idx = array_a[0:id_train]
df_train = df.iloc[train_idx,:]
val_idx = array_a[id_train+1:id_train+id_val]
df_val = df.iloc[val_idx,:]
test_idx = array_a[id_train+id_val+1:-1]
df_test = df.iloc[test_idx,:]

#%%
import matplotlib.image as mpimg
img = mpimg.imread(df_train.path.iloc[0])
plt.imshow(img)
z = torch.zeros(16,28,28)
z_in = torch.tensor(img)
z[1,:,:] = z_in

#%%
t = np.arange(0,len(df_train))
n_batch = 16
num_iter = 1

#%%
import torch.nn.functional as F
def get_batch(df_train,t,n_batch,num_iter,z_path,z_lab):
    k=0
    for i in range(num_iter*n_batch,(num_iter+1)*n_batch):
        img = mpimg.imread(df_train.path[df_train.index[t[i]]])
        z_in = torch.tensor(img)
        z_path[k,:,:] = z_in
        idx = df_train.label[df_train.index[t[i]]]
        idx = torch.tensor(int(idx))
        idx = F.one_hot(idx, num_classes=10)
        z_lab[k,:]=idx
        k+=1
    return z_path, z_lab

z_lab = torch.zeros(n_batch,10)
z_path = torch.zeros(n_batch,28,28)

X,Y = get_batch(df_train, t, n_batch, 100,z_path,z_lab)
#%%
z_test = torch.zeros(len(df_test),28,28)
z_test_lab = torch.zeros(len(df_test),10)
def get_test(df_test,z_test,z_test_lab):
    for i in range(len(df_test)):
        img = mpimg.imread(df_test.path[df_test.index[t[i]]])
        z_in = torch.tensor(img)
        z_test[i,:,:] = z_in
        idx = df_test.label[df_test.index[t[i]]]
        idx = torch.tensor(int(idx))
        idx = F.one_hot(idx, num_classes=10)
        z_test_lab[i,:]=idx
    return z_test, z_test_lab
    
X_test, Y_test = get_test(df_test,z_test,z_test_lab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_test = X_test.to(device=device)
Y_test = Y_test.to(device=device)
    
#%%
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)
import torch.nn as nn
#torch.backends.cudnn.benchmarks = True #cudnn
#torch.backends.cudnn.allow_tf32 #cudnn
#==========================
# добавить слой нормировки по батчу
# добавить еще один слой свертки и сравнить результаты с исходной реализацией

class ff_net(nn.Module):
    def __init__(self):
        super(ff_net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x 
    
model = ff_net()
model.to(device=device)
 
criterion = nn.CrossEntropyLoss()
#%%
#out = model(X)
#loss = criterion(out,Y)
#%%
import torch.optim as optim

NUM_EPOCHS = 10
num_iter = np.floor(len(df_train)/n_batch)
num_iter = int(num_iter)
best_accuracy = 0.0
z_lab = torch.zeros(n_batch,10).to(device=device)
z_path = torch.zeros(n_batch,28,28).to(device=device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for epoch in range(NUM_EPOCHS):
    test_error_count = 0.0
    #for images, labels in iter(data_load):
    for i in range(num_iter):
        # images, labels = next(iter(dataloader))
        images, labels  = get_batch(df_train, t, n_batch, i, z_path, z_lab)
    

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        test_error_count += float(torch.sum((labels == outputs)))
        
       # test_accuracy = 1.0 - float(test_error_count) / float(len(df_train))
        #print('%d: %f' % (epoch, loss.item()))
      #  if test_accuracy > best_accuracy:
            #torch.save(model.state_dict(), BEST_MODEL_PATH)
           # best_accuracy = test_accuracy  
    if epoch % 1 ==0:
        with torch.no_grad():
            out = model(X_test)
            loss_test = F.cross_entropy(out, Y_test)
            print(loss_test.item())
            acc = torch.sum(out.argmax(1) == Y_test.argmax(1))/float(len(df_test))
            print(acc*100)
            
#%%
                
            









        
