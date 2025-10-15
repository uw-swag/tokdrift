from math import atan2, pi
N, * XY = map(int, open(0).read().split())
XY = list(zip(* [iter(XY)] * 2))
for i, (x, y) in enumerate(XY):
    D = sorted(atan2(X - x, Y - y) for j, (X, Y) in enumerate(XY) if j != i)
    D.append(D[0] + 2 * pi)
    ans = 0
    for a, b in zip(D, D[1:]):
        if b - a >= pi:
            ans = (b - a) - pi
    print(f"{(ans / (2 * pi)):.20f}")
