import multiprocessing as mp
import time


def sequential_polynomial_evaluation(coefficients, x):
    result = 0
    for power, coefficient in enumerate(coefficients):
        result += coefficient * (x**power)
    return result


def parallel_polynomial_worker(chunk, x, start_power):
    subtotal = 0
    for offset, coefficient in enumerate(chunk):
        subtotal += coefficient * (x ** (start_power + offset))
    return subtotal


def build_chunks(coefficients, num_processes):
    n = len(coefficients)
    base = n // num_processes
    remainder = n % num_processes

    chunks = []
    start = 0
    for i in range(num_processes):
        size = base + (1 if i < remainder else 0)
        if size == 0:
            continue
        end = start + size
        chunks.append((coefficients[start:end], start))
        start = end
    return chunks


def parallel_polynomial_evaluation(coefficients, x, num_processes):
    chunks = build_chunks(coefficients, num_processes)

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            parallel_polynomial_worker,
            [(chunk, x, start_power) for chunk, start_power in chunks],
        )

    return sum(results)


def compare_evaluation(coefficients, x, num_processes):
    start_time = time.time()
    sequential_result = sequential_polynomial_evaluation(coefficients, x)
    sequential_time = time.time() - start_time

    start_time = time.time()
    parallel_result = parallel_polynomial_evaluation(coefficients, x, num_processes)
    parallel_time = time.time() - start_time

    print(f"Sequential Result: {sequential_result}")
    print(f"Parallel Result: {parallel_result}")
    print(f"Sequential Time: {sequential_time:.6f} seconds")
    print(f"Parallel Time: {parallel_time:.6f} seconds")

    if sequential_time > parallel_time:
        print(f"Parallel evaluation is faster by {sequential_time - parallel_time:.6f} seconds.")
    else:
        print("Sequential evaluation is faster or has similar performance.")


def main():
    # Polynomial with coefficients in ascending power order:
    # 7 + 1x + 5x^2 + 2x^3 + 0x^4 + 3x^5
    coefficients = [7, 1, 5, 2, 0, 3]
    x = 2
    num_processes = 2

    compare_evaluation(coefficients, x, num_processes)


if __name__ == "__main__":
    main()
