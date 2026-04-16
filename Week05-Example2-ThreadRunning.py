import threading
import time


def task():
    print("Task started.")
    time.sleep(2)
    print("Task finished.")


# Running state: thread starts and runs concurrently.
thread = threading.Thread(target=task)
thread.start()
thread.join()
print("Thread has completed execution.")