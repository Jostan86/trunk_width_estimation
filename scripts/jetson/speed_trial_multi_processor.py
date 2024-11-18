import numpy as np
import time
from multiprocessing import Pool, cpu_count

"""Do some speed trials of numpy computations on the CPU with multiple processors, although i'm pretty sure it break"""

# Define the size of the arrays
array_size = 1000000

# Create two large arrays
array1 = np.random.rand(array_size)
array2 = np.random.rand(array_size)

# Define the number of iterations
num_iterations = 1

# Function to perform dot product
def dot_product(_):
    result1 = np.dot(array1, array2)
    return result1

# Function to perform elementwise product
def elementwise_product(_):
    result2 = np.sum(array1 * array2)
    return result2

# Multi-threaded execution
if __name__ == '__main__':
    with Pool(cpu_count()) as p:
        start_time = time.time()
        p.map(dot_product, range(num_iterations))
        dot_product_time_total = time.time() - start_time

        start_time = time.time()
        p.map(elementwise_product, range(num_iterations))
        elementwise_product_time_total = time.time() - start_time

    dot_product_time_avg = dot_product_time_total / num_iterations
    elementwise_product_time_avg = elementwise_product_time_total / num_iterations

    # Report the results to the user
    print(f"Dot product average time: {dot_product_time_avg} seconds")
    print(f"Elementwise product average time: {elementwise_product_time_avg} seconds")
