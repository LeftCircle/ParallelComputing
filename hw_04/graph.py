import matplotlib
# Use Agg backend for non-interactive environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import path_utils
import os

def parse_results_file(filepath):
    """Parse the results file to extract matrix names and chunked SpMV times."""
    matrix_names = []
    chunked_times = []
    c_times = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Matrix:'):
            matrix_name = line.split('Matrix:')[1].strip()
            matrix_names.append(matrix_name)
            
            # Find the C SpMV time
            i += 1
            c_time_match = re.search(r'(\d+\.\d+)', lines[i])
            if c_time_match:
                c_times.append(float(c_time_match.group(1)))
            
            # Find the Chunked SpMV time
            i += 1
            chunked_time_match = re.search(r'(\d+\.\d+)', lines[i])
            if chunked_time_match:
                chunked_times.append(float(chunked_time_match.group(1)))
        i += 1
        
    return matrix_names, chunked_times, c_times

def create_bar_graph(matrix_names, chunked_times):
    """Create a bar graph of chunked SpMV times by matrix."""
    plt.figure(figsize=(12, 6))
    
    # Create bar graph
    bars = plt.bar(matrix_names, chunked_times, color='skyblue', width=0.6)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}s',
                ha='center', va='bottom', rotation=0)
    
    # Add labels and title
    plt.xlabel('Matrix Name')
    plt.ylabel('Chunked SpMV Time (seconds)')
    plt.title('Chunked SpMV Performance by Matrix')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('chunked_spmv_performance.png')
    #plt.show()

def create_comparison_graph(matrix_names, chunked_times, c_times):
    """Create a bar graph comparing C and Chunked SpMV times by matrix."""
    plt.figure(figsize=(14, 7))
    
    # Set position of bars on X axis
    x = np.arange(len(matrix_names))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, c_times, width, label='C Implementation', color='green')
    plt.bar(x + width/2, chunked_times, width, label='Chunked SpMV', color='skyblue')
    
    # Add labels and title
    plt.xlabel('Matrix Name')
    plt.ylabel('Time (seconds) - Log Scale')
    plt.title('Performance Comparison: C vs Chunked SpMV Implementation')
    plt.xticks(x, matrix_names, rotation=45, ha='right')
    plt.legend()
    
    # Use log scale for y-axis due to large differences
    plt.yscale('log')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('spmv_implementation_comparison.png')
    #plt.show()

if __name__ == "__main__":
    # Parse the results file
    filename = path_utils.get_current_folder() + '/results.txt'
    matrix_names, chunked_times, c_times = parse_results_file(filename)
    
    # Create the bar graph for Chunked SpMV times
    create_bar_graph(matrix_names, chunked_times)
    
    # Create a comparison graph (C vs Chunked)
    #create_comparison_graph(matrix_names, chunked_times, c_times)