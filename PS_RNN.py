import numpy as np
import time
import keras
import os
from preprocessing import Preprocessing
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, Callback
from models import Train_LSTM
from sklearn.model_selection import train_test_split

start_time = time.time()

path_BH = os.path.join('data', 'BH')
path_NS = os.path.join('data', 'NS')
preprocessor = Preprocessing() 

bin_factor = 100
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

# Define a custom callback to save the model at epoch 5
class SaveModelAtEpoch5(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == 5:  # Save model at epoch 5
            self.model.save('model_epoch_5.h5')


# Define the model checkpoint callback
model_checkpoint = ModelCheckpoint(filepath='model_checkpoint.h5',
                                    monitor='val_loss',
                                    save_best_only=False,
                                    save_weights_only=False,
                                    mode='auto',
                                    period=1,
                                    verbose=1)

# Combine the two callbacks
callbacks = [model_checkpoint, SaveModelAtEpoch5()]

test_loss, test_acc, prediction  = Train_LSTM(X_train, y_train,
                                              X_val, y_val,
                                              X_test, y_test,
                                              nodes,
                                              batch_size= 30,
                                              num_features = 1,
                                              epochs = 7)

end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds\n")
print(f"That's equal to: {(end_time - start_time)/3600} hours\n")
