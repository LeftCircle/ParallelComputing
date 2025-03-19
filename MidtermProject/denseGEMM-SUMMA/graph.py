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

txt_file = open("results_to_graph.txt", "r")
lines = txt_file.readlines()

# Create a dictionary to store the data
data = {}
for i in range(0, len(lines), 6):
	name = lines[i].strip()
	m = int(lines[i+1].split()[2])
	k = int(lines[i+1].split()[4])
	n = int(lines[i+1].split()[6])
	size = (m, k, n)
	init_time = float(lines[i+2].split()[2])
	comm_time = float(lines[i+3].split()[2])
	comp_time = float(lines[i+4].split()[2])
	total_time = float(lines[i+5].split()[2])
	if name not in data:
		data[name] = {}
	if size not in data[name]:
		data[name][size] = []
	data[name][size].append((init_time, comm_time, comp_time, total_time))

# Create a graph for each name
for name in data:
	color = np.random.rand(3,)
	plt.figure()
	plt.title(name)
	plt.xlabel("Size")
	plt.ylabel("Time (s)")
	for size in data[name]:
		x = [i for i in range(len(data[name][size]))]
		y = [data[name][size][i][3] for i in range(len(data[name][size]))]
		plt.plot(x, y, color=color, label=str(size))
	plt.legend()
	plt.show()

txt_file.close()

