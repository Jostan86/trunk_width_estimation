import numpy as np
import time

"""Do some speed trials of numpy computations on the CPU"""

# Define the size of the arrays
array_size = 1000000

# Create two large arrays
array1 = np.random.rand(array_size)
array2 = np.random.rand(array_size)

# Define the number of iterations
num_iterations = 1000

# Initialize variables to accumulate the differences
dot_product_time_total = 0
elementwise_product_time_total = 0

# Run the operations multiple times
for _ in range(num_iterations):
    start_time = time.time()
    result1 = np.dot(array1, array2)
    end_time = time.time()
    dot_product_time_total += end_time - start_time

    start_time = time.time()
    result2 = np.sum(array1 * array2)
    end_time = time.time()
    elementwise_product_time_total += end_time - start_time

# Calculate the average times
dot_product_time_avg = dot_product_time_total / num_iterations
elementwise_product_time_avg = elementwise_product_time_total / num_iterations

# Report the results to the user
print(f"Dot product average time: {dot_product_time_avg} seconds")
print(f"Elementwise product average time: {elementwise_product_time_avg} seconds")