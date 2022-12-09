
import numpy as np
import numba, time
from numba import cuda

# print(np.__version__)
# print(numba.__version__)
# cuda.detect()

#############
#    EX1    #
#############

# When a kernel is launched it has a grid associated with it. 
# A grid is composed of blocks; 
# A block is composed of threads.

# Example 1.1: Add scalars

# Function to add two scalars
@cuda.jit
def add_scalars(a, b, c):
    c[0] = a + b

# Allocate a 1D array of 1 element
dev_c = cuda.device_array(shape=(1,), dtype=np.float32)

# Call the kernel on the GPU
add_scalars[6, 4](2.0, 7.0, dev_c)   # 6 block, 4 thread

# For printing the result, we need to copy the array from the GPU to the CPU
c = dev_c.copy_to_host()
print(f"2.0 + 7.0 = {c[0]}")   #  2.0 + 7.0 = 9.0


#############
#    EX2    #
#############

# Function to add two arrays
@cuda.jit
def add_array(a, b, c):
    
    # Get the index of the current thread
    i = cuda.grid(1)   # i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    
    # Check if the index is within the array bounds
    if i < a.size:
        
        # Add the elements at the index
        c[i] = a[i] + b[i]

# Sample data
N = 1_000_000
a = np.arange(N, dtype=np.float32)
b = np.arange(N, dtype=np.float32)

# Move the array from host to device
dev_a = cuda.to_device(a)            
dev_b = cuda.to_device(b) 

# Allocate the result array on the device
dev_c = cuda.device_array_like(a)

# Configure the blocks and threads
threads_per_block = 256
blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block

# Start the kernel
t1 = time.time()
add_array[blocks_per_grid, threads_per_block](dev_a, dev_b, dev_c)
t2 = time.time()

# Time report
print(f"Time: {t2 - t1} s")
# For printing the result, we need to copy the array from the GPU to the CPU
c = dev_c.copy_to_host()
print(c)


#############
#    EX3    #
#############

# Example 1.4: Add arrays with grid striding (dynamically allocated blocks)

# Function to add two arrays
@cuda.jit
def add_array_gs(a, b, c):
    
    # Get the index of the current thread as starting point
    i_start = cuda.grid(1)
    
    # Get the number of threads in the grid
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    
    # Loop starting from i_start and incrementing by the number of threads in the grid
    for i in range(i_start, a.size, threads_per_grid):
        
        # Add the elements at the index
        c[i] = a[i] + b[i]

# Sample data
N = 10_000_000
a = np.arange(N, dtype=np.float32)
b = np.arange(N, dtype=np.float32)

# Move the array from host to device
dev_a = cuda.to_device(a)            
dev_b = cuda.to_device(b) 

# Allocate the result array on the device
dev_c = cuda.device_array_like(a)

# Configure the blocks and threads
threads_per_block = 256
blocks_per_grid_gs = 32 * 80  # Use 32 * multiple of streaming multiprocessors

# Start the kernel
add_array_gs[blocks_per_grid_gs, threads_per_block](dev_a, dev_b, dev_c)
cuda.synchronize()

# For printing the result, we need to copy the array from the GPU to the CPU
t1 = time.time()
c = dev_c.copy_to_host()
t2 = time.time()

# Time report
print(f"Time: {t2 - t1} s")







































# import multiprocessing
# from joblib import Parallel, delayed

# x =[1,2,3,4,5]
# y =[3,5,7,8,7]

# def function(val1,val2):
#     return val1 + val2

# result = Parallel(n_jobs=multiprocessing.cpu_count()-3)(
#     delayed(function)(i,j,) for (i,j) in zip(x,y))

# print(result)