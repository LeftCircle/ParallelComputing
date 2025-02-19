import re
import numpy as np

import matplotlib.pyplot as plt
import path_utils as pu

# Get files and paths
current_folder = pu.get_current_folder()
file_paths = pu.get_all_files_of_suffix(current_folder, ".txt")
file_names = [pu.get_file_name_without_extension(path) for path in file_paths]

# Data structures to hold organized data
data_dict = {}  # Dictionary to hold times for each matrix and method
unique_matrices = set()
unique_methods = set()

# Parse each file
for method, file_path in zip(file_names, file_paths):
    with open(file_path, 'r') as f:
        for line in f:
            if '.mtx' in line:
                # Extract matrix name and time
                matrix = re.findall(r'\w+\.mtx', line)[0]
                time = float(re.findall(r'\d+\.\d+', line)[0])
                
                # Store data
                if matrix not in data_dict:
                    data_dict[matrix] = {}
                data_dict[matrix][method] = time
                unique_matrices.add(matrix)
                unique_methods.add(method)

# Convert to lists for plotting
matrices = sorted(list(unique_matrices))
methods = sorted(list(unique_methods))

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Set width of bars and positions of the bars
bar_width = 0.8 / len(methods)
r = np.arange(len(matrices))

# Create bars for each method
for idx, method in enumerate(methods):
    times = [data_dict[matrix].get(method, 0) for matrix in matrices]
    position = r + idx * bar_width
    plt.bar(position, times, bar_width, label=method)

# Customize the plot
plt.xlabel('Matrix Name')
plt.ylabel('Time (ms)')
plt.title('Performance Comparison by Matrix and Method')
plt.xticks(r + bar_width * (len(methods)-1)/2, matrices, rotation=45, ha='right')
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.show()