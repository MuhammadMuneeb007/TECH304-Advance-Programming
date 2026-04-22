import multiprocessing as mp


def worker(shared_value, shared_array):
    shared_value.value += 1
    for i in range(len(shared_array)):
        shared_array[i] += 2


def main():
    shared_value = mp.Value("i", 0)
    shared_array = mp.Array("i", [1, 2, 3, 4, 5])

    process = mp.Process(target=worker, args=(shared_value, shared_array))
    process.start()
    process.join()

    print("Shared Value:", shared_value.value)
    print("Shared Array:", list(shared_array))


if __name__ == "__main__":
    main()
