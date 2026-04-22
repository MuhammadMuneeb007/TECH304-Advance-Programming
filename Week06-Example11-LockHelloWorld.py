import time
from multiprocessing import Lock, Process


def print_hello_world(lock, index):
    lock.acquire()
    try:
        time.sleep(1)
        print("hello world", index)
    finally:
        lock.release()


def main():
    shared_lock = Lock()
    processes = []

    for num in range(10):
        process = Process(target=print_hello_world, args=(shared_lock, num))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
