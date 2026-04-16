import threading


thread_local_data = threading.local()


def worker_with_thread_local():
    thread_local_data.value = threading.current_thread().name
    print(
        f"Thread ID: {threading.current_thread().ident} - "
        f"Thread {thread_local_data.value} is running with its own thread-local data."
    )


worker_threads = [threading.Thread(target=worker_with_thread_local) for _ in range(3)]

for thread in worker_threads:
    thread.start()

for thread in worker_threads:
    thread.join()
