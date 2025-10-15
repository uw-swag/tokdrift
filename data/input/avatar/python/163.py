import collections
import heapq
import sys
import math
import itertools
import bisect
from io import BytesIO, IOBase
import os
def value(): return tuple(map(int, input().split()))
def values(): return tuple(map(int, sys.stdin.readline().split()))
def inlst(): return [int(i) for i in input().split()]
def inlsts(): return [int(i) for i in sys.stdin.readline().split()]
def inp(): return int(input())
def inps(): return int(sys.stdin.readline())
def instr(): return input()
def stlst(): return [i for i in input().split()]


def help(a, b, l):
    tot = []
    for i in range(b):
        tot.append(l[i * a: i * a + a])
    for i in zip(* tot):
        if sum((i)) == b:
            return True
    return False


def solve():
    tot = []
    x = instr()
    s = []
    for i in x:
        if i == 'O':
            s.append(0)
        else:
            s.append(1)
    for i in range(1, 13):
        if 12 % i == 0:
            if help(i, 12 // i, s):
                tot.append((12 // i, i))
    if len(tot) == 0:
        print(len(tot), end='')
    else:
        print(len(tot), end=' ')
    for a, b in sorted(tot):
        if a == sorted(tot)[-1][0] and b == sorted(tot)[-1][1]:
            print(f'{a}x{b}', end='')
        else:
            print(f'{a}x{b}', end=' ')
    print()


if __name__ == "__main__":
    for i in range(inp()):
        solve()
