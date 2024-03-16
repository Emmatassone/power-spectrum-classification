import numpy as np
from preprocessing import Preprocessing
from sklearn.preprocessing import StandardScaler
import keras
from models import Train_LSTM
import os
from preprocessing import Preprocessing
from sklearn.model_selection import train_test_split

path_BH = os.path.join('data', 'BH')
path_NS = os.path.join('data', 'NS')
preprocessor = Preprocessing() 

bin_factor = 10
BH_powerspectra = preprocessor.array_collect(path_BH,
                                             bin_factor = bin_factor,
                                             BH = True)
NS_powerspectra = preprocessor.array_collect(path_NS,
                                             bin_factor = bin_factor,
                                             BH=False)

powerspectra=np.vstack((BH_powerspectra,NS_powerspectra))


nodes = 32768 

X_test = StandardScaler().fit_transform(powerspectra[:,1:2])
y_test = powerspectra[:,3]

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                test_size = 0.66,
                                                random_state = 42)

np.random.shuffle(powerspectra)
X_train = powerspectra[:,1:2]
y_train = powerspectra[:,3]

test_loss, test_acc, prediction  = Train_LSTM(X_train, y_train,
                                              X_val, y_val,
                                              X_test, y_test,
                                              nodes,
                                              batch_size= 30,
                                              num_features = 1,
                                              epochs = 10)
