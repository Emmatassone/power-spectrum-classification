import numpy as np
import os

class Preprocessing:
    def __init__(self): # -> None:
        pass
    # Binning function
    def rebin_file(self, file_path, rebinning_factor=100):
    
        # Read the ASCII file, skipping the first 12 rows
        data = np.loadtxt(file_path, skiprows=12, delimiter=None, usecols=(0, 1, 2))
#        data = np.genfromtxt(file_path, skip_header=12, delimiter=None, usecols=(0, 1, 2))

        # Read the frequency and the power values
        frequency = data[:-1, 0]
        power = data[:-1, 1]
        errors = data[:-1, 2]

        # Calculate the new frequency bin width based on the rebinning factor
        new_bin_width = (frequency[-1] - frequency[0]) / (len(frequency) / rebinning_factor)
        # (frequency[-1] - frequency[0]) calculates the total range of frequencies in the dataset
        # len(frequency) gives the total number of frequency values in the dataset.
        # len(frequency) / rebinning_factor calculates the number of new bins to be created after rebinning.
        
        # Create the new frequency bins
        new_frequency_bins = np.arange(min(frequency), max(frequency) + new_bin_width, new_bin_width)
        num_new_bins = len(new_frequency_bins)
        # np.arange(...) generates a sequence of values starting from the minimum frequency value, increasing by the new_bin_width, and stopping at or just beyond the maximum frequency value. The + new_bin_width is included to ensure that the upper boundary of the last bin covers the maximum frequency value. As a result, new_frequency_bins is an array that contains the boundaries of the new frequency bins, and num_new_bins stores the total number of these new bins.
        
        # Initialise arrays to store re-binned frequency and power values
        rebinned_frequency = np.zeros(num_new_bins)
        rebinned_power = np.zeros(num_new_bins)
        rebinned_errors = np.zeros(num_new_bins)
        
        # Iterate through the new frequency bins and calculate the re-binned power values
        for i in range(num_new_bins - 1):
            bin_start = new_frequency_bins[i]
            bin_end = new_frequency_bins[i + 1]
            
            # Select power values within the current bin
            values_in_bin = power[(frequency >= bin_start) & (frequency < bin_end)]
            errors_in_bin = errors[(frequency >= bin_start) & (frequency < bin_end)]
            
            # Calculate the weighted mean power value within the bin
            weighted_mean_power = np.sum(values_in_bin / (errors_in_bin**2)) / np.sum(1 / (errors_in_bin**2))
            
            # Calculate the weighted mean error within the bin
            weighted_mean_error = 1 / np.sqrt(np.sum(1 / (errors_in_bin**2)))
            
            # Calculate the mean power value within the bin
            #    rebinned_power[i] = np.mean(values_in_bin)
            
            # Store the corresponding frequency value
            rebinned_frequency[i] = (bin_start + bin_end) / 2
            rebinned_power[i] = weighted_mean_power
            rebinned_errors[i] = weighted_mean_error
            
            rebinned_data = np.column_stack((rebinned_frequency, rebinned_power, rebinned_errors))
#            output_file = file_path.split("/")[-1] + ".rebinned" + rebinning_factor
            output_file = "output.asc" + ".rebinned" + rebinning_factor
            np.savetxt(output_file, rebinned_data, header="   Frequency     Power    Errors")
            
            return rebinned_data
        
    def collect_all_powerspectra(self, object_path, bin_factor=100, BH=True):
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
                        tmp_spectrum_file = self.rebin_file(tmp_spectrum_file, bin_factor)
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
