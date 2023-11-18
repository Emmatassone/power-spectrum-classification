import numpy as np
import pandas as pd
from preprocessing import Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import keras
import matplotlib.pyplot as plt
from models import jordi_CNN,nodes_CNN       
import os
from preprocessing import Preprocessing

path_BH = os.path.join('data', 'BH')
path_NS = os.path.join('data', 'NS')
preprocessor = Preprocessing() 

bin_factor=10
BH_powerspectra = preprocessor.array_collect(path_BH, bin_factor=bin_factor, BH=True)
NS_powerspectra = preprocessor.array_collect(path_NS, bin_factor=bin_factor, BH=False)
powerspectra=np.vstack((BH_powerspectra,NS_powerspectra))

nodes=32768#powerspectra[:,:,0:2].shape[1]

X_test = powerspectra[:,0:2].reshape(-1,nodes,2)
y_test = powerspectra[:,3]

one_hot_test = keras.utils.to_categorical(y_test, num_classes = 2)

# Reshape to (8 * 32768, 3), shuffle, and reshape back to (8, 32768, 3)
np.random.shuffle(powerspectra)
X_train = powerspectra[:,0:2]
y_train = powerspectra[:,3]

one_hot_train = keras.utils.to_categorical(y_train, num_classes = 2)

#standarize for CNN
X_train=StandardScaler().fit_transform(X_train)
#Reshape to in CNN format
X_train=X_train.reshape(-1,nodes,2)


#model=jordi_CNN(X_train, one_hot_train.reshape((-1,2)), nodes)
model=jordi_CNN(X_train, one_hot_train.reshape((-1,2)), nodes)

prediction = model.predict(X_test, batch_size=1)

#test_loss, test_acc = model.evaluate(X_test, np.mean(one_hot_test.reshape(-1,nodes,2),axis=1), batch_size=1)
test_loss, test_acc = model.evaluate(X_test, one_hot_test, batch_size=1)