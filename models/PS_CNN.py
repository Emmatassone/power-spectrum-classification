import numpy as np
import pandas as pd
from preprocessing import Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import keras
import matplotlib.pyplot as plt
from models import train_CNN       
import os
from preprocessing import Preprocessing

path_BH = os.path.join('data', 'BH')
path_NS = os.path.join('data', 'NS')

preprocessor = Preprocessing(path_BH, path_NS)
powerspectra = preprocessor.collect_all_NS_BH_data()

nodes = 32768
#nodes = preprocessor.shape[1]?
rows = preprocessor.shape[0]

shuffled_PS = np.random.shuffle(powerspectra)

#model=jordi_CNN(X_train, one_hot_train.reshape((-1,2)), nodes)
model = train_CNN(X_train = shuffled_PS[ : , 1:2 ], y_train = shuffled_PS[ : , 3 ],
                X_val = powerspectra[ :rows//2 , 1:2 ], y_val = powerspectra[ :rows//2 , 3 ],
                epochs = 1,
                batch_size = 30,
                )


prediction = model.predict(X_test, batch_size=1)

test_loss, test_acc = model.evaluate(X_test, np.mean(one_hot_test.reshape(-1,nodes,2),axis=1), batch_size=1)
#test_loss, test_acc = model.evaluate(X_test, one_hot_test, batch_size=1)
