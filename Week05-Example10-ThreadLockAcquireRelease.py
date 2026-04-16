import threading


counter = 0
lock = threading.Lock()


def increment():
    global counter
    print(f"Thread {threading.current_thread().name} attempting to acquire the lock...")

    lock.acquire()
    print(f"Thread {threading.current_thread().name} acquired the lock.")

    current_value = counter
    counter = current_value + 1
    print(f"Thread {threading.current_thread().name} incremented the counter to {counter}.")

    lock.release()
    print(f"Thread {threading.current_thread().name} released the lock.")


threads = []
for i in range(5):
    t = threading.Thread(target=increment, name=f"Thread-{i}")
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Final counter value: {counter}")
