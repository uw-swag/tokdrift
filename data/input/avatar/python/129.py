import math
given = input("")
l1 = given.split()
l1 = [int(x) for x in l1]
x1 = l1[0]
y1 = l1[1]
x2 = l1[2]
y2 = l1[3]
denominator = x2 - x1
numerator = y2 - y1
if denominator != 0:
    quotient = numerator / denominator
if numerator == 0:
    d = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    x4 = x1
    x3 = x2
    y3 = y2 + d
    y4 = y1 + d
    print(f"{x4} {y4} {x3} {y3}")
elif denominator == 0:
    y4 = y2
    y3 = y1
    d = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    x4 = x1 + d
    x3 = x2 + d
    print(f"{x3} {y3} {x4} {y4}")
elif quotient == 1:
    x4 = x2
    x3 = x1
    y4 = y1
    y3 = y2
    print(f"{x3} {y3} {x4} {y4}")
elif quotient == - 1:
    x4 = x1
    x3 = x2
    y4 = y2
    y3 = y1
    print(f"{x4} {y4} {x3} {y3}")
else:
    print('-1')
