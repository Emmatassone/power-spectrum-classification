from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Reshape
from tensorflow.keras.callbacks import Callback
import tensorflow as tf


def jordi_CNN(X_train, one_hot_train, nodes,epochs=10, batch_size=64,validation_split=0.05):
    model = Sequential([
        Conv1D(input_shape=(X_train.shape[1], X_train.shape[2]), filters=128,
                               kernel_size=32,
                               strides=2, use_bias=True,
                               activation=tf.nn.relu),
        Dropout(rate=0.8),
        Flatten(),
        Dense(64, activation=tf.nn.relu),
        Dense(2, activation=tf.nn.softmax)
        ])
    adam = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999,epsilon=1e-08)
    # Compile and train netowrk
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, one_hot_train, epochs=epochs, batch_size=batch_size,validation_split=validation_split, shuffle=True)
    
    return model

def nodes_CNN(X_train, y_train, nodes,epochs=10, batch_size=64,validation_split=0.05):
    model = Sequential([
        Conv1D(input_shape=(X_train.shape[1], X_train.shape[2]), filters=128,
                               kernel_size=32,
                               strides=2, use_bias=True,
                               activation=tf.nn.relu),
        Dropout(rate=0.8),
        Flatten(),
        Dense(64, activation=tf.nn.relu),
        Dense(nodes, activation='sigmoid')  # Output one value per frequency
        #Reshape((nodes, 1))  # Reshape to 
        ])
    adam = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999,epsilon=1e-08)
    # Compile and train netowrk
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=validation_split, shuffle=True)
    
    return model