from multiprocessing import Process, Queue

def worker_function(output_queue):
    output_queue.put([100, None, 'Queue'])

if __name__ == '__main__':
    result_queue = Queue()
    worker_process = Process(target=worker_function, args=(result_queue,))
    worker_process.start()
    worker_process.join()
    print(result_queue.get(0))