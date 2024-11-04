import numpy as np
import time
from models import train_CNN       
import os
from termcolor import colored
from preprocessing.preprocessing import Preprocessing
import argparse

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Train an Convolutional Neural Network model with specified arguments.")
    
    # Add arguments
    parser.add_argument('--batch_size', type=str, required=True, help='Batch size for for CNN')
    parser.add_argument('--epochs', type=str, required=True, help='Number of iterations for CNN')
    
    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments in your RNN training logic
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)

    print(colored(f"Running RNN with batch size = {BATCH_SIZE} and number of epochs={EPOCHS}","red"))
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

    _ , _ , test_accuracy = train_CNN(
            X_train = powerspectra[ : , :, 0:2 ], y_train = np.mean(powerspectra[ :, : , 2 ], axis=1).reshape(-1,1),
            X_val = ps[ :NUM_FILES//2 , :, 0:2 ], y_val = np.mean(ps[ :NUM_FILES//2 , : , 2 ], axis=1).reshape(-1,1),
            X_test = ps[ NUM_FILES//2: , :, 0:2 ], y_test =  np.mean( ps[ NUM_FILES//2:, : , 2 ], axis=1).reshape(-1,1),
            batch_size = BATCH_SIZE,
            epochs = EPOCHS,
            save_interval = EPOCHS
            )
       
    end_time = time.time()
    
    computing_time=(end_time - start_time)/3600
    print("Total time taken: {} hours".format(round(computing_time,2)))
    accuracy_filename = './models/metrics/CNN_accuracies.txt'
    with open(accuracy_filename, 'a') as f:
        f.write(f"Accuracy on the test set: {test_accuracy:.4f}\n")
        f.write(f"run node number : {NODES}\n")
        f.write(f"run feature number : {NUM_FEATURES}\n")
        f.write(f"run batch size : {BATCH_SIZE}\n")
        f.write(f"run epoch number: {EPOCHS}\n\n")

if __name__ == "__main__":
    main()
