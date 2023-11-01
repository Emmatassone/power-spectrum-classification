import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import Callback
from preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

class PlotValidationLoss(Callback):
    def __init__(self,bin_factor):
        self.bin_factor=bin_factor
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
    
        val_loss = logs.get('val_loss')
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
        self.val_losses.append(val_loss)
        if (epoch+1) % 15 == 0 or epoch == 0:
            self.plot_validation_loss()

    def plot_validation_loss(self):
        epochs_range = range(0, len(self.val_losses))
        plt.plot(epochs_range, self.val_losses, 'b', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.savefig('Training-CNN_Validation_Error_'+str(self.bin_factor)+'.png')
        
start_time = time.time()

path_BH = os.path.join('data', 'BH')
path_NS = os.path.join('data', 'NS')
preprocessor = Preprocessing() 

bin_factor=100

BH_powerspectra = preprocessor.collect_all_powerspectra(path_BH, bin_factor=bin_factor, BH=True)
NS_powerspectra = preprocessor.collect_all_powerspectra(path_NS, bin_factor=bin_factor, BH=False)

data_array=np.vstack([BH_powerspectra,NS_powerspectra])
data=pd.DataFrame(data_array,columns=['freq','power','error','BH?'])

post_processing_time = time.time()

X = data[['freq', 'power']]
y = data['BH?']

# Assuming 'X' is your input data
# Calculate mean and standard deviation
mean = np.mean(X, axis=0)
std_dev = np.std(X, axis=0)

# Standardize the data
X_standardized = (X - mean) / std_dev
# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Assuming 'X' is your input data and 'y' is the corresponding target labels
# Make sure to appropriately preprocess your data (e.g., normalization)

# Define your model
model = Sequential()

sequence_length, num_features=len(X),3
# Add 1D convolutional layer
model.add(Conv1D(filters=16, kernel_size=128, activation='relu', input_shape=(sequence_length, num_features)))

# Add max pooling layer
model.add(MaxPooling1D(pool_size=2))

# Add additional convolutional layers or other layers as needed
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))

# Flatten the output of the last convolutional layer
model.add(Flatten())

# Add fully connected layers
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))  # Dropout for regularization

# Output layer for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

plot_validation_loss = PlotValidationLoss(bin_factor)
# Train the model
epochs,batch_size=150, 64
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)

# Make predictions
predictions = model.predict(X_test)

# Optionally, save the model
model.save('CNN_model_'+str(bin_factor)+'.h5')

# Save the test accuracy in a text file
with open('CNN_model_'+str(bin_factor)+'_test_accuracy_and_parameters.txt', 'w') as f:
    f.write(f'Test Accuracy: {test_acc:.4f} \n')
    f.write('____________________________\n')
    f.write('Model Architecture:\n\n')
    model.summary(print_fn=lambda x: f.write(x + '\n'))