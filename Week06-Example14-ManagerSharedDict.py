import multiprocessing as mp


def worker(shared_dict):
    shared_dict["counter"] += 1
    shared_dict["values"].append(shared_dict["counter"])


def main():
    with mp.Manager() as manager:
        shared_dict = manager.dict()
        shared_dict["counter"] = 0
        shared_dict["values"] = manager.list([1])

        processes = [mp.Process(target=worker, args=(shared_dict,)) for _ in range(10)]
        for process in processes:
            process.start()

        for process in processes:
            process.join()

        print("Shared Dictionary Values:", list(shared_dict["values"]))


if __name__ == "__main__":
    main()
