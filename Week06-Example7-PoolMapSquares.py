import multiprocessing
import os
import time


def my_function(arg):
    print(f"Process ID: {os.getpid()} is processing argument: {arg}")
    time.sleep(1)
    result = arg**2
    print(f"Process ID: {os.getpid()} finished processing argument: {arg}")
    return result


def main():
    my_list_of_args = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    with multiprocessing.Pool(processes=4) as pool:
        result = pool.map(my_function, my_list_of_args)

    print(f"Results: {result}")
    print("All processes finished execution.")


if __name__ == "__main__":
    main()
