import multiprocessing as mp
import time

import numpy as np


def matrix_multiply_worker(task):
    row_index, row, matrix_b = task
    result_row = np.dot(row, matrix_b)
    return row_index, result_row


def parallel_matrix_multiply(matrix_a, matrix_b, pool_size):
    m, n = matrix_a.shape
    n2, p = matrix_b.shape
    if n != n2:
        raise ValueError("Matrix dimensions do not match for multiplication.")

    with mp.Pool(pool_size) as pool:
        tasks = [(i, matrix_a[i, :], matrix_b) for i in range(m)]
        results = pool.map(matrix_multiply_worker, tasks)

    matrix_c = np.zeros((m, p))
    for row_index, result_row in results:
        matrix_c[row_index, :] = result_row
    return matrix_c


def measure_execution_time(matrix_a, matrix_b, pool_sizes):
    timings = {}
    for pool_size in pool_sizes:
        start_time = time.time()
        _ = parallel_matrix_multiply(matrix_a, matrix_b, pool_size)
        timings[pool_size] = time.time() - start_time
    return timings


def main():
    matrix_a = np.random.rand(400, 300)
    matrix_b = np.random.rand(300, 200)

    pool_sizes = [1, 2, 4]
    timings = measure_execution_time(matrix_a, matrix_b, pool_sizes)

    print("Execution Time Analysis:")
    for pool_size, elapsed_time in timings.items():
        print(f"Pool size: {pool_size}, Time: {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    main()
