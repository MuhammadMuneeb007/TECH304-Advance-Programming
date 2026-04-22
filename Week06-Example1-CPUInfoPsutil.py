import psutil


def main():
    logical_cpus = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    cpu_freq = psutil.cpu_freq()

    print(f"Logical CPUs (threads): {logical_cpus}")
    print(f"Physical Cores: {physical_cores}")
    if cpu_freq is not None:
        print(f"CPU Frequency: {cpu_freq.current:.2f} MHz")
    else:
        print("CPU Frequency: Unknown")


if __name__ == "__main__":
    main()
