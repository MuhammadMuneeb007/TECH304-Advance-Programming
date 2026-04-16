import _thread
import time


def thread_function_1():
    thread_id = _thread.get_ident()
    print(f"Thread 1 ID: {thread_id} - Performing task: Printing message 1")
    time.sleep(2)
    print(f"Thread 1 ID: {thread_id} - Task completed.")


def thread_function_2():
    thread_id = _thread.get_ident()
    print(f"Thread 2 ID: {thread_id} - Performing task: Printing message 2")
    time.sleep(3)
    print(f"Thread 2 ID: {thread_id} - Task completed.")


def main():
    try:
        _thread.start_new_thread(thread_function_1, ())
        _thread.start_new_thread(thread_function_2, ())
        time.sleep(5)
        print("Main thread is done waiting, program complete.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
