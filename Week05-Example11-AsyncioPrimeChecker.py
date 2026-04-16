import asyncio
import math


async def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    limit = int(math.sqrt(n)) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True


async def check_prime(n, prime_numbers):
    if await is_prime(n):
        prime_numbers.append(n)


async def find_primes_in_range(start, end):
    prime_numbers = []
    tasks = [asyncio.create_task(check_prime(n, prime_numbers)) for n in range(start, end + 1)]
    await asyncio.gather(*tasks)
    return prime_numbers


async def main():
    start = int(input("Enter the start number of the range: "))
    end = int(input("Enter the end number of the range: "))

    prime_numbers = await find_primes_in_range(start, end)
    print(f"Prime numbers between {start} and {end}: {prime_numbers}")


asyncio.run(main())
