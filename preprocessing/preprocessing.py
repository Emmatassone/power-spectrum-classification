import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class Preprocessing:
    def __init__(self, path_BH, path_NS):
        self.path_BH = path_BH
        self.path_NS = path_NS
        pass        
    
    def array_collect(self, object_path, bin_factor=100, BH=True):
        result_arrays_list = []
        for source in os.listdir(object_path):
            for observation in os.listdir(os.path.join(object_path,source)):
                observation_path = os.path.join(object_path, source, observation, 'pca')
                # Check if observation_path is a directory
                if os.path.isdir(observation_path):
                    
                    # List all rebinned .asc files in the observation_path
                    list_rebinned_PS = [file for file in os.listdir(observation_path) if file.endswith('.asc_' + str(bin_factor))]
                    # Iterate over each .asc file
                    for spectrum_file in list_rebinned_PS:
                        binned_powerspectra_file=os.path.join(observation_path, spectrum_file)
                        tmp_spectrum_file = np.loadtxt(binned_powerspectra_file)#, skiprows=12
                        # Add a new target column to write down whether we have a BH or not
                        self.nodes = tmp_spectrum_file.shape[0]
                        new_column = np.ones(self.nodes) if BH else np.zeros(self.nodes)
                        new_column_reshaped = new_column.reshape(-1, 1)
                        result_array = np.hstack((tmp_spectrum_file, new_column_reshaped))
                        result_arrays_list.append(result_array)
                    else:
                        continue
                else:
                    continue
        # Convert the list of result arrays into a single NumPy array
        return np.vstack(result_arrays_list)
    
    def collect_all_NS_BH_data(self, bin_factor=100):
        BH_powerspectra = self.array_collect(self.path_BH, 
                                             bin_factor = bin_factor,
                                             BH = True
                                             )
        NS_powerspectra = self.array_collect(self.path_NS,
                                             bin_factor = bin_factor,
                                             BH = False
                                             )
        
        
        powerspectra = np.vstack((BH_powerspectra,NS_powerspectra))
        self.shape = powerspectra.shape
#        powerspectra_scaled = np.hstack([powerspectra[:, 0], StandardScaler().fit_transform(powerspectra[:, 1]), powerspectra[:, 3]])
        print("Spectra succesfully collected")
        return powerspectra.reshape(-1,self.nodes,4)