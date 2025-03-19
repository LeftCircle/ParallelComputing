# Help me create a graph for data in the following format:
# name
# m k n size
# Initialization time
# Communication time
# Computation time
# Total time

# Where each name has the same color, and there are sepoerate lines for 
# each different m k n size

# stationary_c_summa
# m = 32768, k = 128, n = 32768 with grid size 8
# Initialization time: 0.115426
# Communication time: 1.204734
# Computation time: 0.323193
# Total time: 1.643468
# stationary_A_summa
# m = 4096, k = 4096, n = 4096 with grid size 8
# Initialization time: 0.457262
# Communication time: 0.384781
# Computation time: 0.223644
# Total time: 1.066642

# The data is just like above, in a text file



import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
results_file = os.path.join(script_dir, "results_to_graph.txt")

txt_file = open(results_file, "r")
lines = txt_file.readlines()
lines = [line.strip() for line in lines if line.strip()]

# Create a dictionary to store the data
data = {}
# The x axis should be m, k, n. The y axis should be total time.
# The color should be the name of the algorithm
for i in range(0, len(lines), 6):
	name = lines[i].strip()
	# Pull just the values for m, k, and n from 
	# # m = 32768, k = 128, n = 32768 with grid size 8, 
	dim_line = lines[i+1].strip()
	dimensions = re.findall(r'\d+', dim_line)
	m = int(dimensions[0])
	k = int(dimensions[1])
	n = int(dimensions[2])

	#m, k, n, size = map(int, lines[i+1].split(" ")[3:])
	print(m, k, n)
	init_time = float(lines[i+2].split(": ")[1])
	comm_time = float(lines[i+3].split(": ")[1])
	comp_time = float(lines[i+4].split(": ")[1])
	total_time = float(lines[i+5].split(": ")[1])

	# if name not in data:
	# 	data[name] = {"m": [], "k": [], "n": [], "size": [], "total_time": []}
	# data[name]["m"].append(m)
	# data[name]["k"].append(k)
	# data[name]["n"].append(n)
	# data[name]["total_time"].append(total_time)
	if name not in data:
			data[name] = {
				"dims": [],
				"init_time": [],
				"comm_time": [],
				"comp_time": [],
				"total_time": []
			}
		
	dim_label = f"{m}×{k}×{n}"
	data[name]["dims"].append(dim_label)
	data[name]["init_time"].append(init_time)
	data[name]["comm_time"].append(comm_time)
	data[name]["comp_time"].append(comp_time)
	data[name]["total_time"].append(total_time)

# # Create the graph
# fig, ax = plt.subplots()
# for name in data:
# 	ax.plot(data[name]["m"], data[name]["total_time"], label=name)
# plt.xlabel("m")
# plt.ylabel("Total Time")
# plt.legend()
# plt.show()
 # Create the grouped bar graph
fig, ax = plt.subplots(figsize=(12, 6))

# Set width of bars and positions
bar_width = 0.15
opacity = 0.8

# Colors for different timing components
# colors = {
# 	"init_time": "#2ecc71",    # green
# 	"comm_time": "#e74c3c",    # red
# 	"comp_time": "#3498db",    # blue
# 	"total_time": "#9b59b6"    # purple
# }

colors = {
	"stationary_c_summa": "#2ecc71",    # green
	"stationary_A_summa": "#e74c3c",    # red
}

# Create bars for each algorithm and timing type
for idx, name in enumerate(data):
	x = np.arange(len(data[name]["dims"]))
	
	# Create grouped bars for each timing component
	for i, metric in enumerate(["init_time", "comm_time", "comp_time", "total_time"]):
		if metric is "total_time":
			position = x + (i + idx * len(colors)) * bar_width
			plt.bar(position, 
					data[name][metric],
					bar_width,
					alpha=opacity,
					color=colors[name],
					label=f"{name} - {metric}")

# Customize the plot
plt.xlabel('Matrix Dimensions (m×k×n)')
plt.ylabel('Time (seconds)')
plt.title('SUMMA Performance Comparison')

# # Set x-axis labels
# all_dims = []
# for name in data:
# 	all_dims.extend(data[name]["dims"])
# unique_dims = sorted(list(set(all_dims)))

# plt.xticks(np.arange(len(unique_dims)) + bar_width * 4,
# 			unique_dims,
# 			rotation=45,
# 			ha='right')

# After collecting all dimensions in all_dims:
all_dims = []
for name in data:
    all_dims.extend(data[name]["dims"])

# Function to check if dimensions are square
def is_square(dim_str):
    m, k, n = map(int, dim_str.replace('×', ' ').split())
    return m == k == n

# Function to get size for sorting
def get_size(dim_str):
    m, k, n = map(int, dim_str.replace('×', ' ').split())
    return max(m, k, n)

# Separate and sort square and rectangular dimensions
square_dims = sorted([d for d in set(all_dims) if is_square(d)], key=get_size)
rect_dims = sorted([d for d in set(all_dims) if not is_square(d)], key=get_size)

# Combine sorted dimensions
unique_dims = square_dims + rect_dims

# Update x-axis ticks
plt.xticks(np.arange(len(unique_dims)) + bar_width * 4,
           unique_dims,
           rotation=45,
           ha='right')

# Add a vertical line to separate square and rectangular matrices
if square_dims and rect_dims:
    plt.axvline(x=(len(square_dims) - 0.5), color='gray', linestyle='--', alpha=0.5)


plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


txt_file.close()

