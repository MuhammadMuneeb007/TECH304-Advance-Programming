import multiprocessing as mp
import os
import time


def my_fun(arg1, arg2):
    child_pid = os.getpid()
    print(f"Child Process ID: {child_pid}")
    print(f"Child Process started with arguments: {arg1}, {arg2}")
    result = arg1 + arg2
    time.sleep(20)
    print(f"Child Process Result: {result}")
    print(f"Child Process ID: {child_pid} - Execution finished.")


def main():
    parent_pid = os.getpid()
    print(f"Parent Process ID: {parent_pid}")

    val1 = 5
    val2 = 10

    process = mp.Process(target=my_fun, args=(val1, val2))
    print("Starting child process...")
    process.start()
    process.join()

    print("Parent process completed.")


if __name__ == "__main__":
    main()
