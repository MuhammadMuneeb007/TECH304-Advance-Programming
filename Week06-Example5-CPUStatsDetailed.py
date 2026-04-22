import platform

import psutil


def get_cpu_stats():
    logical_cpus = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)

    try:
        from cpuinfo import get_cpu_info

        cpu_info = get_cpu_info()
        processor_model = cpu_info.get("brand_raw", "Unknown")
        processor_vendor = cpu_info.get("vendor_id_raw", "Unknown")
    except ImportError:
        processor_model = "Unknown (install py-cpuinfo)"
        processor_vendor = "Unknown (install py-cpuinfo)"

    cpu_freq = psutil.cpu_freq()
    current_freq = cpu_freq.current if cpu_freq else None
    max_freq = cpu_freq.max if cpu_freq else None

    system_name = platform.system()
    machine_arch = platform.machine()
    os_version = platform.version()

    if max_freq is not None:
        computational_power = f"{logical_cpus * max_freq:.2f} MHz"
    else:
        computational_power = "Unknown"

    print("=== CPU Stats ===")
    print(f"Processor Model: {processor_model}")
    print(f"Processor Vendor: {processor_vendor}")
    print(f"Number of Logical CPUs (Threads): {logical_cpus}")
    print(f"Number of Physical Cores: {physical_cores}")
    print(
        "Current Frequency:",
        f"{current_freq:.2f} MHz" if current_freq is not None else "Unknown",
    )
    print(
        "Maximum Frequency:",
        f"{max_freq:.2f} MHz" if max_freq is not None else "Unknown",
    )
    print(f"System: {system_name}")
    print(f"Machine Architecture: {machine_arch}")
    print(f"OS Version: {os_version}")
    print(f"Estimated Computational Power: {computational_power}")


if __name__ == "__main__":
    get_cpu_stats()
