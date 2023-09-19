#! /usr/bin/python
# -*- coding: utf-8 -*-       #Indicamos tipo de codificación

import numpy as np
import pandas as pd
from stingray import Powerspectrum

#############################################################################
#                      1st way of rebinning                                 #
#############################################################################

# Define the path to your ASCII file
file_path = "96442-01-01-00_FS37_E_125.asc"

# Read the ASCII file, skipping the first 12 rows
data = np.genfromtxt(file_path, skip_header=12, delimiter=None, usecols=(0, 1, 2))

# Read the frequency and the power values
frequency = data[:-1, 0]
power = data[:-1, 1]
errors = data[:-1, 2]

# Define the rebinning factor
rebinning_factor = 50

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


# Now, rebinned_frequency, rebinned_power, and rebinned_errors contain the linearly re-binned data with a factor of 50.

# Now we save the re-binned data in a file
# Define the output file path
output_file = "rebinned_power_spectrum.dat"

# Create a list of lists to hold the rebinned data
rebinned_data = []
#rebinned_data.append(["   Frequency", "   Power", "   Errors"])
header_row = [f"{'Frequency':>15}", f"{'Power':>10}", f"{'Errors':>12}"]
rebinned_data.append(header_row)

whitespaces = 3
# Append the rebinned data to the list
for i in range(num_new_bins):
    rebinned_data.append([f"{' ' * whitespaces}{rebinned_frequency[i]}", rebinned_power[i], rebinned_errors[i]])

# Calculate the maximum width for each column
max_widths = [max(len(str(item)) for item in col) for col in zip(*rebinned_data)]

# Save the data to a tab-separated file
with open(output_file, 'w') as file:
    for line in rebinned_data:
        formatted_line = '\t'.join(f"{item:{width}}" for item, width in zip(line, max_widths))
        file.write(formatted_line + '\n')
#        file.write('\t'.join(map(str, line)) + '\n')


print("")
print(f"Rebinned data saved to {output_file}")
print("")
print("Don't forget to take into account that for some reason, the last raw is 0")



#############################################################################
#                 2nd way of rebinning (using Stringray)                    #
#############################################################################

# Create a Powerspectrum object
#power_spectrum = Powerspectrum()
#power_spectrum.freq = your_frequency_array  # Replace with your frequency values
#power_spectrum.power = your_power_array  # Replace with your power values

