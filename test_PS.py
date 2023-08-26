""" ***************************************************************************
# * File Description:                                                         *
# * Workflow for model building                                               *
# * --------------------------------------------------------------------------*
# * AUTHORS(S): Frank Ceballos <frank.ceballos89@gmail.com>                   *
# * --------------------------------------------------------------------------*
# * DATE CREATED: June 26, 2019                                               *
# * --------------------------------------------------------------------------*
# * NOTES: None                                                               *
# * ************************************************************************"""

###############################################################################
#                          1. Importing Libraries                             #
###############################################################################
# Visual bar for data generation
#from tqdm.notebook import tqdm
import pickle
import os

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Metrics
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Binning function
def rebin_PS(data, bin_factor):
    num_rows, num_cols = data.shape
    new_num_rows = num_rows // bin_factor
    
    # Reshape the data for binning
    reshaped_data = data[:new_num_rows * bin_factor, :].reshape(new_num_rows, bin_factor, num_cols)
    
    # Calculate the mean along the specified axis (axis=1)
    rebinned_data = np.mean(reshaped_data, axis=1)
    
    return rebinned_data

bin_factor = 30

path_BH = './data_test/BH/'
path_NS = './data_test/NS/'
result_arrays_list = []
for source in os.listdir(path_BH):
    for observation in os.listdir(path_BH+source):
        observation_path = os.path.join(path_BH, source, observation, 'pca/')

        # Check if observation_path is a directory
        if os.path.isdir(observation_path):
            
            # Check if there are any .asc files in the observation_path
            list_power_spectra = [file for file in os.listdir(observation_path) if file.endswith('.asc')]
            for spectrum_file in list_power_spectra:  # Iterate over each file
                tmp_spectrum_file = np.loadtxt(observation_path+spectrum_file, skiprows=12)
#                temp_df = pd.read_csv(path_BH+observation+'/pca/'+energy_spectrum, skiprows=12, names=('freq','power','error'), delim_whitespace=True, dtype={'freq': np.float64, 'power': np.float64, 'error': np.float64})
#                temp_df=pd.read_fwf(path_BH+observation+'/pca/'+energy_spectrum,skiprows=12, names=('freq','power','error'))
                tmp_spectrum_file=rebin_PS(tmp_spectrum_file, bin_factor)

                # Create a new column to add
                new_column = np.ones(tmp_spectrum_file.shape[0])

                # Reshape array2 to have shape (32768, 1)
                new_column_reshaped = new_column.reshape(-1, 1)

                # Concatenate array2 as a new column to array1
                result_array = np.hstack((tmp_spectrum_file, new_column_reshaped))
                
                # Assuming result_array is obtained in each iteration
                result_arrays_list.append(result_array)
#                temp_df['BH?']=1

            else:
                print("No .asc file(s) found in observation folder:", observation_path)
        else:
            print("Skipping non-directory:", observation_path)

# Convert the list of result arrays into a single NumPy array
BH_stacked_result_array = np.vstack(result_arrays_list)

print(BH_stacked_result_array.shape)


result_arrays_list = []
for source in os.listdir(path_NS):
    for observation in os.listdir(path_NS+source):
        observation_path = os.path.join(path_NS, source, observation, 'pca/')

        # Check if observation_path is a directory
        if os.path.isdir(observation_path):
            
            # Check if there are any .asc files in the observation_path
            list_power_spectra = [file for file in os.listdir(observation_path) if file.endswith('.asc')]
            for spectrum_file in list_power_spectra:  # Iterate over each file
                tmp_spectrum_file = np.loadtxt(observation_path+spectrum_file, skiprows=12)
#                temp_df = np.loadtxt(path_NS+source+'/'+observation+'/pca/'+spectrum_file, skiprows=12)
#                temp_df = pd.read_fwf(os.path.join(observation_path, spectrum_file), skiprows=12, names=('freq','power','error'))
                tmp_spectrum_file=rebin_PS(tmp_spectrum_file, bin_factor)

                # Create a new column to add
                new_column = np.zeros(tmp_spectrum_file.shape[0])

                # Reshape array2 to have shape (32768, 1)
                new_column_reshaped = new_column.reshape(-1, 1)

                # Concatenate array2 as a new column to array1
                result_array = np.hstack((tmp_spectrum_file, new_column_reshaped))
                
                # Assuming result_array is obtained in each iteration
                result_arrays_list.append(result_array)
#                temp_df['BH?']=1

            else:
                print("No .asc file(s) found in observation folder:", observation_path)
        else:
            print("Skipping non-directory:", observation_path)

# Convert the list of result arrays into a single NumPy array
NS_stacked_result_array = np.vstack(result_arrays_list)

print(NS_stacked_result_array.shape)

