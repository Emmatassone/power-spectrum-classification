#! /usr/bin/python
# -*- coding: utf-8 -*-       # Indicamos tipo de codificación

import numpy as np
import pandas as pd
from stingray import Powerspectrum

#############################################################################
#                   1st way of logarithmic rebinning                        #
#############################################################################

# Define the path to your ASCII file
file_path = "96442-01-01-00_FS37_E_125.asc"

# Read the ASCII file, skipping the first 12 rows
data = np.genfromtxt(file_path, skip_header=12, delimiter=None, usecols=(0, 1, 2))

# Read the frequency and the power values along with their errors
frequency = data[:-1, 0]
power = data[:-1, 1]
errors = data[:-1, 2]

# Define the rebinning factor for logarithmic rebinning
log_rebinning_factor = 1.1

# Calculate the new frequency bins logarithmically
new_frequency_bins = [frequency[0]]
while new_frequency_bins[-1] < frequency[-1]:
    new_frequency_bins.append(new_frequency_bins[-1] * log_rebinning_factor)

# Convert the list of new frequency bins to a numpy array
new_frequency_bins = np.array(new_frequency_bins)

# Calculate the corresponding bin widths
bin_widths = new_frequency_bins[1:] - new_frequency_bins[:-1]

# Initialise arrays to store re-binned frequency and power values
num_new_bins = len(new_frequency_bins) - 1
rebinned_frequency = np.zeros(num_new_bins)
rebinned_power = np.zeros(num_new_bins)
rebinned_errors = np.zeros(num_new_bins)

# Iterate through the new frequency bins and calculate the re-binned power values
for i in range(num_new_bins):
    bin_start = new_frequency_bins[i]
    bin_end = new_frequency_bins[i + 1]

    # Select power values within the current bin
    values_in_bin = power[(frequency >= bin_start) & (frequency < bin_end)]
    errors_in_bin = errors[(frequency >= bin_start) & (frequency < bin_end)]

    # Calculate the weighted mean power value within the bin
    weighted_mean_power = np.sum(values_in_bin / (errors_in_bin ** 2)) / np.sum(1 / (errors_in_bin ** 2))

    # Calculate the weighted mean error within the bin
    weighted_mean_error = 1 / np.sqrt(np.sum(1 / (errors_in_bin ** 2)))

    # Store the corresponding frequency value
    rebinned_frequency[i] = (bin_start + bin_end) / 2
    rebinned_power[i] = weighted_mean_power
    rebinned_errors[i] = weighted_mean_error

# Now, rebinned_frequency, rebinned_power, and rebinned_errors contain the logarithmically re-binned data.

# Save the re-binned data in a file
# Define the output file path
output_file = "rebinned_power_spectrum.log"

# Create a list of lists to hold the rebinned data
rebinned_data = []
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

print("")
print(f"Logarithmically rebinned data saved to {output_file}")
print("")
