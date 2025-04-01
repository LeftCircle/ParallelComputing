import matplotlib.pyplot as plt
import numpy as np
import re

results_file = 'results.txt'

def parse_results(filename):
    k_values = []
    reference_times = []
    blocked_times = []
    simd_times = []
    simd_omp_times = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if 'k:' in line:
            k = int(re.search(r'k: (\d+)', line).group(1))
            k_values.append(k)
        elif 'Reference:' in line:
            time = int(re.search(r'Reference:\s+(\d+)ms', line).group(1))
            reference_times.append(time)
        elif 'Blocked:' in line:
            time = int(re.search(r'Blocked:\s+(\d+)ms', line).group(1))
            blocked_times.append(time)
        elif 'SIMD:' in line and 'OMP' not in line:
            time = int(re.search(r'SIMD:\s+(\d+)ms', line).group(1))
            simd_times.append(time)
        elif 'SIMD+OMP:' in line:
            time = int(re.search(r'SIMD\+OMP:\s+(\d+)ms', line).group(1))
            simd_omp_times.append(time)
    
    return k_values, reference_times, blocked_times, simd_times, simd_omp_times

def create_bar_graph(k_values, reference_times, blocked_times, simd_times, simd_omp_times):
    x = np.arange(len(k_values))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width*1.5, reference_times, width, label='Reference')
    rects2 = ax.bar(x - width/2, blocked_times, width, label='Blocked')
    rects3 = ax.bar(x + width/2, simd_times, width, label='SIMD')
    rects4 = ax.bar(x + width*1.5, simd_omp_times, width, label='SIMD+OMP')
    
    ax.set_ylabel('Time (ms)')
    ax.set_xlabel('K Value')
    ax.set_title('Performance Comparison of Different Implementations')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend()
    
    # Add value labels on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', rotation=90)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    plt.tight_layout()
    #plt.savefig('performance_comparison.png')
    plt.show()

def main():
    k_values, reference_times, blocked_times, simd_times, simd_omp_times = parse_results(results_file)
    create_bar_graph(k_values, reference_times, blocked_times, simd_times, simd_omp_times)

if __name__ == "__main__":
    main()