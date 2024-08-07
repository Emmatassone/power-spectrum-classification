from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Reshape, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

def jordi_CNN(X_train, one_hot_train, nodes, epochs=6, batch_size=64, validation_split=0.05):
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
    adam = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # Compile and train network
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, one_hot_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=True)

    return model

def nodes_CNN(X_train, y_train, nodes, epochs=6, batch_size=64, validation_split=0.05):
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
    adam = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # Compile and train network
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=validation_split,
              shuffle=True)

    return model

def Train_LSTM(X_train, y_train, X_val, y_val, X_test, y_test,
               time_steps, batch_size=5, num_features=1, epochs=6, load=False):
    # Model architecture
    if load == False:

        model = Sequential([
            LSTM(units=64, input_shape=(time_steps, num_features)),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

    else:
        model = load_model('model_epoch_06.h5')

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Define the model checkpoint callback
    checkpoint_path = "model_epoch_{epoch:02d}.h5"
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=False,
                                  save_freq='epoch',
                                  verbose=1,
                                  period=2)  # Save every 2 epochs

    history = model.fit(X_train, y_train,
                        epochs = epochs,
                        batch_size = batch_size,
                        validation_data = (X_val, y_val),
                        callbacks = [cp_callback])

    test_loss, test_acc = model.evaluate(X_test, y_test)
#    predictions = model.predict(X_test)
    return  test_loss, test_acc  #, predictions
