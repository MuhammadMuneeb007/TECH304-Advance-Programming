import threading
import time


event = threading.Event()


def worker_wait_for_event():
    print(f"Thread ID: {threading.current_thread().ident} - Worker waiting for event to be set...")
    event.wait()
    print(f"Thread ID: {threading.current_thread().ident} - Worker received the event!")


def trigger_event():
    print(f"Thread ID: {threading.current_thread().ident} - Triggering event...")
    time.sleep(2)
    event.set()


worker_thread = threading.Thread(target=worker_wait_for_event)
trigger_thread = threading.Thread(target=trigger_event)

worker_thread.start()
trigger_thread.start()

worker_thread.join()
trigger_thread.join()
