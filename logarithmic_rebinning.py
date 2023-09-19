#! /usr/bin/python
# -*- coding: utf-8 -*-       # Indicamos tipo de codificación

import numpy as np
import pandas as pd
from stingray import Powerspectrum

#############################################################################
#                     Logarithmic rebinning                                 #
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
log_rebinning_factor = 30

# Initialise arrays to store re-binned frequency and power values
rebinned_frequency = []
rebinned_power = []
rebinned_errors = []

# Initialise variables to keep track of the current bin
bin_start = frequency[0]
bin_end = bin_start * 10**(1 / log_rebinning_factor)
bin_power_sum = 0
bin_error_sum = 0
num_frequencies_in_bin = 0

# Iterate through the frequency values and calculate the re-binned power values
for i in range(len(frequency)):
    if frequency[i] <= bin_end:
        # Add the power to the current bin
        bin_power_sum += power[i]
        bin_error_sum += errors[i]**2  # Sum of squared errors (to be divided by num_frequencies_in_bin later)
        num_frequencies_in_bin += 1
    else:
        # Calculate the average power for the current bin
        if num_frequencies_in_bin > 0:
            avg_power = bin_power_sum / num_frequencies_in_bin
            avg_error = np.sqrt(bin_error_sum) / num_frequencies_in_bin  # Propagate the error correctly
            rebinned_frequency.append((bin_start + bin_end) / 2)
            rebinned_power.append(avg_power)
            rebinned_errors.append(avg_error)

        # Move to the next bin
        bin_start = bin_end
        bin_end = bin_start * 10**(1 / log_rebinning_factor)
        bin_power_sum = 0
        bin_error_sum = 0
        num_frequencies_in_bin = 0

# Handle the last bin if it is not complete
if num_frequencies_in_bin > 0:
    avg_power = bin_power_sum / num_frequencies_in_bin
    avg_error = np.sqrt(bin_error_sum) / num_frequencies_in_bin  # Propagate the error correctly
    rebinned_frequency.append((bin_start + bin_end) / 2)
    rebinned_power.append(avg_power)
    rebinned_errors.append(avg_error)

# Now, rebinned_frequency and rebinned_power contain the logarithmically re-binned data.

# Save the re-binned data in a file
# Define the output file path
output_file = "rebinned_power_spectrum.log"

# Create a list of lists to hold the rebinned data
rebinned_data = []
header_row = [f"{'Frequency':>15}", f"{'Power':>10}", f"{'Errors':>12}"]
rebinned_data.append(header_row)

whitespaces = 3
# Append the rebinned data to the list
for i in range(len(rebinned_frequency)):
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
