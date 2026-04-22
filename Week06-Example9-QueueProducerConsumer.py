from multiprocessing import Process, Queue


def producer(queue):
    for i in range(5):
        queue.put(i)
        print(f"Producer added: {i}")


def consumer(queue):
    while True:
        item = queue.get()
        if item == "DONE":
            break
        print(f"Consumer got: {item}")


def main():
    queue = Queue()

    producer_process = Process(target=producer, args=(queue,))
    consumer_process = Process(target=consumer, args=(queue,))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    queue.put("DONE")
    consumer_process.join()


if __name__ == "__main__":
    main()
