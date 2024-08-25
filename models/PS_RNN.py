import numpy as np
import time
import os
from preprocessing.preprocessing import Preprocessing
from models.models import train_LSTM

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

start_time = time.time()

path_BH = os.path.join('data', 'BH')
path_NS = os.path.join('data', 'NS')

preprocessor = Preprocessing(path_BH, path_NS)
powerspectra = preprocessor.collect_all_NS_BH_data()



#delete error column if there is one
powerspectra  = np.delete(powerspectra , 2, axis=2)

NODES = preprocessor.nodes
NUM_FEATURES = powerspectra.shape[2]-1
NUM_FILES = powerspectra.shape[0]

ps=np.copy(powerspectra)

np.random.shuffle(powerspectra)
#test_loss, test_acc, prediction = Train_LSTM(X_train, y_train,
train_LSTM(X_train = powerspectra[ : , :, 0:2 ], y_train = np.mean(powerspectra[ :, : , 2 ], axis=1).reshape(-1,1),
            X_val = ps[ :NUM_FILES//2 , :, 0:2 ], y_val = np.mean(ps[ :NUM_FILES//2 , : , 2 ], axis=1).reshape(-1,1),
            X_test = ps[ NUM_FILES//2: , :, 0:2 ], y_test =  np.mean( ps[ NUM_FILES//2:, : , 2 ], axis=1).reshape(-1,1),
            time_steps=NODES,
            batch_size=5,
            num_features=NUM_FEATURES,
            epochs=2)

end_time = time.time()
print("Total time taken: {} seconds\n".format(round(end_time - start_time,2)))
print("That's equal to: {} hours\n".format(round((end_time - start_time)/3600,2)))
