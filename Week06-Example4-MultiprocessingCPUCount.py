import multiprocessing as mp


def main():
    print("Number of CPUs:", mp.cpu_count())


if __name__ == "__main__":
    main()
