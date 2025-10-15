from math import *


def next_int():
    return int(input())


def next_ints():
    return list(map(int, input().split()))


n = next_int()
t = 0
for i in range(n):
    if i + 1 & 1:
        t += 1
print(f"{(t / n):.10f}")
