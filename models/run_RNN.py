import numpy as np
import time
import os
from preprocessing.preprocessing import Preprocessing
from models import train_LSTM
from termcolor import colored
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Train an Recurrent Neural Network model with specified arguments.")
    
    # Add arguments
    parser.add_argument('--batch_size', type=str, required=True, help='Batch size for for RNN')
    parser.add_argument('--epochs', type=str, required=True, help='Number of iterations for RNN')
    
    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments in your RNN training logic
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)

    print(colored(f"\nRunning RNN with batch size = {BATCH_SIZE} and number of epochs = {EPOCHS}\n","red"))
    start_time = time.time()
    
    path_BH = os.path.join('data', 'BH')
    path_NS = os.path.join('data', 'NS')
    
    preprocessor = Preprocessing(path_BH, path_NS)
    powerspectra = preprocessor.collect_all_NS_BH_data()
    
    powerspectra = np.delete(powerspectra , 2, axis=2)
    
    NODES = preprocessor.nodes
    NUM_FEATURES = powerspectra.shape[2]-1
    NUM_FILES = powerspectra.shape[0]
    
    np.random.shuffle(powerspectra)
    
    train_end = int(0.8 * NUM_FILES)
    val_end = int(0.9 * NUM_FILES)

    # Split the data
    X_train = powerspectra[:train_end, :, 0:2]
    y_train = np.mean(powerspectra[:train_end, :, 2], axis=1).reshape(-1, 1)

    X_val = powerspectra[train_end:val_end, :, 0:2]
    y_val = np.mean(powerspectra[train_end:val_end, :, 2], axis=1).reshape(-1, 1)

    X_test = powerspectra[val_end:, :, 0:2]
    y_test = np.mean(powerspectra[val_end:, :, 2], axis=1).reshape(-1, 1)
    
    # Train the LSTM model
    _, _, test_accuracy = train_LSTM(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            time_steps=NODES,
            batch_size=BATCH_SIZE,
            num_features=NUM_FEATURES,
            epochs=EPOCHS
        )

        
    end_time = time.time()
    computing_time=(end_time - start_time)/3600
    print(colored("\nTotal time taken: {} hours\n".format(round(computing_time,2)), "green"))

    accuracy_filename = './models/metrics/RNN_accuracies.txt'
    with open(accuracy_filename, 'a') as f:
        f.write(f"Accuracy on the test set: {test_accuracy:.4f}\n")
        f.write(f"run timesteps number : {NODES}\n")
        f.write(f"run feature number : {NUM_FEATURES}\n")
        f.write(f"run batch size : {BATCH_SIZE}\n")
        f.write(f"run epoch number: {EPOCHS}\n\n")
  
if __name__ == "__main__":
    main()


