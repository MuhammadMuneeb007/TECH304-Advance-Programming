import _thread
import time


def thread_task_1():
	print("Thread 1 started")
	time.sleep(10)
	print("Thread 1 finished")


def thread_task_2():
	print("Thread 2 started")
	time.sleep(1.5)
	print("Thread 2 finished")


_thread.start_new_thread(thread_task_1, ())
_thread.start_new_thread(thread_task_2, ())

# Keep main thread alive so child threads can finish.
time.sleep(30)
