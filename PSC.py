#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os

def run_script(script_name, args):
        # Ensure PYTHONPATH includes the root directory
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get('PYTHONPATH', '')

    command = [sys.executable, script_name] + args
    result = subprocess.run(command, env=env)  # Pass the modified environment
    if result.returncode != 0:
        print(f"Error occurred while running {script_name}")
    return result

def main():
    parser = argparse.ArgumentParser(description='Run model training scripts with specified arguments.')
    
    # Add mutually exclusive group to choose the model
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--RNN', action='store_true', help='Train the Recurrent Neural Network model')
    group.add_argument('--CNN', action='store_true', help='Train the Convolutional Neural Network model')
    group.add_argument('--RF', action='store_true', help='Train the Random Forest model')
    
    # Add arguments specific to each model
    parser.add_argument('--batch_size', type=str, help='Batch size to use in the Neural Network Model')
    parser.add_argument('--epochs', type=str, help='Number of iterations to do in the Neural Network')
    parser.add_argument('--n_jobs', type=str, help='Number of threads to use in the RF training')
    parser.add_argument('--n_estimators', type=str, help='Number of estimators in the RF')
    parser.add_argument('--min_samples_leaf', type=str, help='Number of samples in the leaf node')
    parser.add_argument('--min_samples_split', type=str, help='Number of samples in the split node')

    args = parser.parse_args()
    
    common_args = []
    if args.batch_size is not None:
        common_args.append(f'--batch_size={args.batch_size}')
    if args.epochs is not None:
        common_args.append(f'--epochs={args.epochs}')
    if args.n_jobs is not None:
        common_args.append(f'--n_jobs={args.n_jobs}')
    if args.n_estimators is not None:
        common_args.append(f'--n_estimators={args.n_estimators}')
    if args.min_samples_leaf is not None:
        common_args.append(f'--min_samples_leaf={args.min_samples_leaf}')
    if args.min_samples_split is not None:
        common_args.append(f'--min_samples_split={args.min_samples_split}')
        
    # Determine which model to run based on the flags
    if args.RNN:
        print(f"Ejecutando RNN con argumentos: {common_args}")
        run_script('models/run_RNN.py', common_args)
    elif args.CNN:
        print(f"Ejecutando CNN con argumentos: {common_args}")
        run_script('models/run_CNN.py', common_args)
    elif args.RF:
        print(f"Ejecutando RF con argumentos: {common_args}")
        run_script('models/run_RF.py', common_args)

if __name__ == "__main__":
    main()
