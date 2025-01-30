import re
import numpy as np

import matplotlib.pyplot as plt

# Given data in the following format in a text file:
# Rank 0: Message size 32768 bytes, Mean latency = 0.025447 ms, Stddev = 0.001174 ms
# Rank 2: Message size 32768 bytes, Mean latency = 0.033758 ms, Stddev = 0.011055 ms
# Rank 4: Message size 32768 bytes, Mean latency = 0.027770 ms, Stddev = 0.003948 ms
# Rank 6: Message size 32768 bytes, Mean latency = 0.024328 ms, Stddev = 0.004062 ms

# Rank 0: Message size 65536 bytes, Mean latency = 0.039056 ms, Stddev = 0.005712 ms
# Rank 2: Message size 65536 bytes, Mean latency = 0.049239 ms, Stddev = 0.008271 ms
# Rank 4: Message size 65536 bytes, Mean latency = 0.042919 ms, Stddev = 0.005877 ms
# Rank 6: Message size 65536 bytes, Mean latency = 0.037983 ms, Stddev = 0.003764 ms

# Rank 0: Message size 131072 bytes, Mean latency = 0.067713 ms, Stddev = 0.004132 ms
# Rank 2: Message size 131072 bytes, Mean latency = 0.074111 ms, Stddev = 0.011579 ms
# Rank 4: Message size 131072 bytes, Mean latency = 0.069535 ms, Stddev = 0.005568 ms
# Rank 6: Message size 131072 bytes, Mean latency = 0.061658 ms, Stddev = 0.005431 ms
#
# Create a plot where the x-axis is the message size and the y-axis is the mean latency.
# and the error bars are the standard deviation.
# Read and parse the data
ranks = []
sizes = []
latencies = []
stddevs = []
n_pairs = 8
fp = r'E:\Programming\NCSU\ParallelComputing\hw_01\p1_results_3.txt'

with open(fp, 'r') as f:
	for line in f:
		match = re.search(r'Rank (\d+): Message size (\d+) bytes, Mean latency = ([\d.]+) ms, Stddev = ([\d.]+) ms', line)
		if match:
			ranks.append(int(match.group(1)))
			sizes.append(int(match.group(2)))
			latencies.append(float(match.group(3)))
			stddevs.append(float(match.group(4)))

# Create a bar plot with a bar for each rank where the height of the bar is the mean latency
# and the error bars are the standard deviation.

# Convert to numpy arrays for easier manipulation
ranks = np.array(ranks)
sizes = np.array(sizes)
latencies = np.array(latencies)
stddevs = np.array(stddevs)


# Get unique sizes and ranks
unique_sizes = np.unique(sizes)
unique_ranks = np.unique(ranks)

unique_sizes = np.sort(unique_sizes)

max_size = sizes.max()
min_size = sizes.min()

max_size_diff = sizes.max()/sizes.min()
max_latencies_diff = latencies.max()/latencies.min()

speed_diff_over_size = max_latencies_diff / max_size_diff

sorted_latencies = np.sort(latencies)
average_latencies = np.zeros(len(sorted_latencies) // n_pairs)
for i in range(0, len(sorted_latencies) - 1, n_pairs):
	# Average each 4 values
	avg = np.mean(sorted_latencies[i:i+n_pairs])
	average_latencies[i//n_pairs] = avg

for i in range(0, len(average_latencies), 2):
	# Get the higher value over the lower one
	higher_over_lower = average_latencies[i+1] / average_latencies[i]
	# only print up to 2 decimal places
	print(f"Higher latency over lower = {higher_over_lower:.2f} for {unique_sizes[i]} bytes and {unique_sizes[i+1]} bytes")
	
#print("Average latencies = ", average_latencies)

print("Max size diff = ", max_size_diff)
print("Max latencies diff = ", max_latencies_diff)
print("Speed diff over size = ", speed_diff_over_size)



# Set width of bars and positions of the bars
width = 0.15 #unique_sizes.min() * 0.2 / unique_sizes.max()
x = np.arange(len(unique_sizes))

print("Width = ", width)
print("X = ", x)

# Create the bar plot
plt.figure(figsize=(10, 6)) 
for i, rank in enumerate(unique_ranks):
	mask = ranks == rank
	plt.bar(x + i*width, latencies[mask], width, 
			label=f'Pair {rank}', yerr=stddevs[mask], capsize=5)

plt.xlabel('Message Size (bytes)')
plt.ylabel('Mean Latency (ms)')
plt.title('Message Latency by Size and Rank')
plt.xticks(x + width*len(unique_ranks)/2, unique_sizes)
plt.legend()
plt.grid(True)
plt.show()