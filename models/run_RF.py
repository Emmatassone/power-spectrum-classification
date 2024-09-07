import numpy as np
from preprocessing.preprocessing import Preprocessing
from models import train_RF
import time
import os
from termcolor import colored
import argparse

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Train a Random Forest model with specified arguments.")
    
    # Add arguments
    parser.add_argument('--batch_size', type=str, required=True, help='Batch size for RF')
    parser.add_argument('--n_jobs', type=str, help='Number of threads to use in the training')
    
    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments in your RNN training logic
    BATCH_SIZE = int(args.batch_size)
    N_JOBS = int(args.n_jobs) if args.n_jobs else 1

    # Example usage
    print(colored(f"Running Ranfom Forest Model with batch size ={BATCH_SIZE} and number of threads={N_JOBS}","red"))
    start_time = time.time()
    
    path_BH = os.path.join('data', 'BH')
    path_NS = os.path.join('data', 'NS')
    
    preprocessor = Preprocessing(path_BH, path_NS)
    powerspectra = preprocessor.collect_all_NS_BH_data(table_format=True)
    
    #delete error column if there is one
    powerspectra  = np.delete(powerspectra , 2, axis=1)
    
    NUM_FILES = powerspectra.shape[0]
    
    ps=np.copy(powerspectra)
    
    np.random.shuffle(powerspectra)
    print(powerspectra[ :,2 ].shape)
    train_RF(   
            X_train = powerspectra[ : , 0:2 ], y_train = powerspectra[ :,2 ],
            X_val = ps[ :NUM_FILES//2 , 0:2 ], y_val = ps[ :NUM_FILES//2 , 2 ],
            X_test = ps[ NUM_FILES//2: , 0:2 ], y_test =  ps[ NUM_FILES//2:, 2 ],
            batch_size = BATCH_SIZE,
            n_jobs = N_JOBS
            )
    
    end_time = time.time()
    
    computing_time=(end_time - start_time)/3600
    print("Total time taken: {} hours".format(round(computing_time,2)))

if __name__ == "__main__":
    main()
