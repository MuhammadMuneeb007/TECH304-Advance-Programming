import threading
import time


def task():
    print("Task started, waiting for 10 seconds...")
    time.sleep(10)
    print("Task completed.")


# Blocked state: worker thread is blocked while sleeping.
thread = threading.Thread(target=task)
thread.start()

for loop in range(1, 10):
    print(loop)

thread.join()
