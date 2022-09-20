import timeit
import numpy as np

N = 100_000_000

def while_loop(n=N):
    i = 0
    s = 0
    while i<n:
        s += i
        i += 1
    return s

def for_loop(n=N):
    s = 0
    for i in range(n):
        s += i
    return s

def sum_range(n=N):
    return sum(range(n))

def sum_numpy(n=N):
    return np.sum(np.arange(n))

def main():
    print("While loop:", timeit.timeit(while_loop, number=1))
    print("For loop:", timeit.timeit(for_loop, number=1))
    print("Sum:", timeit.timeit(sum_range, number=1))
    print("Numpy:", timeit.timeit(sum_numpy, number=1))

if __name__ == "__main__":
    main()