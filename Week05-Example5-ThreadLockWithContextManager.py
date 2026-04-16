import threading


counter = 0
lock = threading.Lock()


def increment():
    global counter
    with lock:
        current_value = counter
        counter = current_value + 1
        print(f"Thread ID: {threading.current_thread().ident} - Counter: {counter}")


threads = [threading.Thread(target=increment) for _ in range(5)]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print(f"Final counter value: {counter}")
