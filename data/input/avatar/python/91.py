def task4(A, B, n):
    for X in range(1001):
        if A * (X ** n) == B:
            return X
    return "No solution"


[A, B, n] = input().split()
print(task4(int(A), int(B), int(n)))
