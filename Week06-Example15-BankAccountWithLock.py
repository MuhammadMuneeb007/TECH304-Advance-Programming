import multiprocessing as mp
import random
import time


def deposit(balance, amount, lock, repeats):
    for _ in range(repeats):
        with lock:
            current_balance = balance.value
            time.sleep(random.uniform(0, 0.05))
            balance.value = current_balance + amount
            print(f"Deposited {amount}, new balance: {balance.value}")


def withdraw(balance, amount, lock, repeats):
    for _ in range(repeats):
        with lock:
            current_balance = balance.value
            time.sleep(random.uniform(0, 0.05))
            balance.value = current_balance - amount
            print(f"Withdrew {amount}, new balance: {balance.value}")


def main():
    lock = mp.Lock()
    balance = mp.Value("d", 1000.0)

    deposit_process = mp.Process(target=deposit, args=(balance, 100, lock, 5))
    withdraw_process = mp.Process(target=withdraw, args=(balance, 50, lock, 5))

    deposit_process.start()
    withdraw_process.start()

    deposit_process.join()
    withdraw_process.join()

    print(f"Final balance: {balance.value}")


if __name__ == "__main__":
    main()
