import numpy as np
from preprocessing.preprocessing import Preprocessing
from models import train_RF
import time
import os
from termcolor import colored
import argparse
import joblib
from sklearn.metrics import accuracy_score

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Train a Random Forest model with specified arguments.")
    
    # Add arguments
    parser.add_argument('--batch_size', type=str, required=True, help='Batch size for RF')
    parser.add_argument('--n_jobs', type=str, help='Number of threads to use in the training')
    parser.add_argument('--n_estimators', type=str, help='Number of estimators in the RF')
    parser.add_argument('--min_samples_leaf', type=str, help='Number of samples in the leaf node')
    parser.add_argument('--min_samples_split', type=str, help='Number of samples in the split node')
    
    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments in your RF training logic
    BATCH_SIZE = int(args.batch_size)
    N_JOBS = int(args.n_jobs) if args.n_jobs else 1
    N_ESTIMATORS = int(args.n_estimators) if args.n_estimators else 200
    MIN_SAMPLES_LEAF = int(args.min_samples_leaf) if args.min_samples_leaf else 20
    MIN_SAMPLES_SPLIT = int(args.min_samples_split) if args.min_samples_split else 50

    # Example usage
    print(colored(f"\nRunning Random Forest Model with batch size={BATCH_SIZE}, number of threads={N_JOBS}\n", "red"))
    start_time = time.time()
    
    path_BH = os.path.join('data', 'BH')
    path_NS = os.path.join('data', 'NS')
    before_processing_time=time.time()
    preprocessor = Preprocessing(path_BH, path_NS)
    powerspectra = preprocessor.collect_all_NS_BH_data(table_format=True)
    after_processing_time = time.time()
    
    print(colored("\nTime taken to process the data: {} seconds\n".format(round(after_processing_time-before_processing_time, 2)), "green"))
    
    # Delete error column if there is one
    powerspectra = np.delete(powerspectra, 2, axis=1)
    
    NUM_FILES = powerspectra.shape[0]
    
    ps = np.copy(powerspectra)
    
    np.random.shuffle(powerspectra)
    print(powerspectra[:, 2].shape)
    
    # Train the Random Forest model
    print("\nCalling train_RF...\n")
    model = train_RF(   
        X_train=powerspectra[:, 0:2],
        y_train=powerspectra[:, 2],
        X_val=ps[:NUM_FILES//2, 0:2], 
        y_val=ps[:NUM_FILES//2, 2],
        X_test=ps[NUM_FILES//2:, 0:2], 
        y_test=ps[NUM_FILES//2:, 2],
        batch_size=BATCH_SIZE,
        n_jobs=N_JOBS,
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT
    )
    
    # Save the model
    model_filename = './models/trained_models/RF_n_estimators-'+str(N_ESTIMATORS)\
                                            +'_min_samples_leaf-'+str(MIN_SAMPLES_LEAF)\
                                            +'_min_samples_split-'+str(MIN_SAMPLES_SPLIT)\
                                            +'_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")
    
    # Evaluate the model
    y_test_pred = model.predict(ps[NUM_FILES//2:, 0:2])
    accuracy = accuracy_score(ps[NUM_FILES//2:, 2], y_test_pred)
    accuracy_filename = './models/metrics/RF_accuracies.txt'
    with open(accuracy_filename, 'a') as f:
        f.write(f"Accuracy on the test set: {accuracy:.4f}\n")
        f.write(f"run n_jobs : {N_JOBS}\n")
        f.write(f"run n_estimators: {N_ESTIMATORS}\n")
        f.write(f"run min_samples_leaf: {MIN_SAMPLES_LEAF}\n")
        f.write(f"run min_samples_split: {MIN_SAMPLES_SPLIT}\n\n")

    end_time = time.time()
    computing_time = (end_time - start_time) / 3600
    print(colored("\n Total time taken: {} hours\n".format(round(computing_time, 2)), "green"))

if __name__ == "__main__":
    main()
