import functools

@functools.lru_cache(maxsize=10)
def factorial(n):
    if n<2:
        return 1
    print(f"calculating factorial({n})")
    return n * factorial(n-1)

if __name__ == "__main__":
    for i in range(2,6):
        print(f"Factorial({i}) = {factorial(i)}")
        print("_" * 50)