import numpy as np
import time
import os
from preprocessing import Preprocessing
from models import train_LSTM

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

start_time = time.time()

path_BH = os.path.join('data', 'BH')
path_NS = os.path.join('data', 'NS')

preprocessor = Preprocessing(path_BH, path_NS)
powerspectra = preprocessor.collect_all_NS_BH_data()

nodes = 32768
#nodes = preprocessor.shape[1]?
rows = preprocessor.shape[0]

shuffled_PS = np.random.shuffle(powerspectra)

#test_loss, test_acc, prediction = Train_LSTM(X_train, y_train,
train_LSTM(X_train = shuffled_PS[ : , 1:2 ], y_train = shuffled_PS[ : , 3 ],
            X_val = powerspectra[ :rows//2 , 1:2 ], y_val = powerspectra[ :rows//2 , 3 ],
            X_test = powerspectra[ rows//2: , 1:2 ], y_test = powerspectra[ rows//2: , 3 ],
            nodes = nodes,
            batch_size = 30,
            num_features = 1,
            epochs = 1,
            load = True)

end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds\n")
print(f"That's equal to: {(end_time - start_time)/3600} hours\n")
