from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Reshape, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class SaveModelEveryNEpochs(tf.keras.callbacks.Callback):
    def __init__(self, save_interval=2, filepath_template="model_epoch_{epoch:02d}.h5"):
        super(SaveModelEveryNEpochs, self).__init__()
        self.save_interval = save_interval
        self.filepath_template = filepath_template

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_interval == 0:
            filepath = self.filepath_template.format(epoch=epoch + 1)
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
            
def model_predictions(model_path,X_new):
    model = load_model(model_path)
    y_pred_prob = model.predict(X_new)
    y_pred = (y_pred_prob > 0.5).astype(int)
    # Evaluate the model on the test set
    return y_pred

def train_CNN(X_train, y_train, X_val, y_val, epochs=6, batch_size=64):
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
    model.fit(X_train, y_train,
            validation_data = (X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        
    return model

def train_LSTM(X_train, y_train, X_val, y_val, X_test, y_test,
               time_steps, batch_size=5, num_features=1, epochs=6, load=False,
               save_interval=2):

    if load:
        # Model architecture
        model = Sequential([
            LSTM(units=64, input_shape=(time_steps, num_features)),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

    else:
        model = load_model('model_epoch_02.h5')
        
        
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Define the model checkpoint callback
    save_callback = SaveModelEveryNEpochs(save_interval=save_interval)

    model.fit(X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[save_callback])


    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test Set loss: ",test_loss,", Test Set accuracy: ", test_acc)
    return model

def train_RF(X_train, y_train, X_val, y_val, batch_size=64):
    X_train=pd.DataFrame(X_train,columns=['freq','power'])
    y_train=pd.DataFrame(y_train,columns=['BH?'])
    X_val=pd.DataFrame(X_val,columns=['freq','power'])
    y_val=pd.DataFrame(y_val,columns=['BH?'])
    X_test, y_test=pd.DataFrame(X_test,columns=['freq','power']),pd.DataFrame(y_test,columns=['BH?'])
    model = RandomForestClassifier(n_estimators=200,
                                   min_samples_leaf=20,
                                   min_samples_split=50,
                                   n_jobs=30
                                  )
    model.fit(X_train, y_train)
    return model
    