n = int(input())
i, j = 1, n * n
while i < j:
    if i % ((n + 1) // 2) == 0:
        print(i, j)
    else:
        print(i, j, end=' ')
    i += 1
    j -= 1
