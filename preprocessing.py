import numpy as np
import os

class Preprocessing:
    def __init__(self): # -> None:
        pass
    # Binning function
    def rebin_PS(self,data, bin_factor):
        num_rows, num_cols = data.shape
        new_num_rows = num_rows // bin_factor

        # Reshape the data for binning
        reshaped_data = data[:new_num_rows * bin_factor, :].reshape(new_num_rows, bin_factor, num_cols)

        # Calculate the mean along the specified axis (axis=1)
        rebinned_data = np.mean(reshaped_data, axis=1)

        return rebinned_data
    
    def collect_all_powerspectra(self, object_path, bin_factor=1, BH=True):
        result_arrays_list = []
        for source in os.listdir(object_path):
            for observation in os.listdir(object_path+source):
                observation_path = os.path.join(object_path, source, observation, 'pca/')

                # Check if observation_path is a directory
                if os.path.isdir(observation_path):
                    
                    # List all .asc files in the observation_path
                    list_power_spectra = [file for file in os.listdir(observation_path) if file.endswith('.asc')]
                    # Iterate over each .asc file
                    for spectrum_file in list_power_spectra:  
                        tmp_spectrum_file = np.loadtxt(observation_path+spectrum_file, skiprows=12)
                        # Rebin the powerspectra 
                        tmp_spectrum_file=self.rebin_PS(tmp_spectrum_file, bin_factor)
                        # Add a new target column to write down whether we have a BH or not
                        new_column = np.ones(tmp_spectrum_file.shape[0]) if BH else np.zeros(tmp_spectrum_file.shape[0])
                        new_column_reshaped = new_column.reshape(-1, 1)
                        result_array = np.hstack((tmp_spectrum_file, new_column_reshaped))
                        result_arrays_list.append(result_array)
                    else:
                        continue
                else:
                    continue
        # Convert the list of result arrays into a single NumPy array
        return np.vstack(result_arrays_list)
