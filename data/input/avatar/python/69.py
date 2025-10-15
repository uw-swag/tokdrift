import sys
import itertools


def solve(N: int, x: "List[int]", y: "List[int]"):
    indices = [i for i in range(N)]
    permutations = itertools.permutations(indices)
    distances = []
    for permutation in permutations:
        total_distance = 0
        for i in range(0, N - 1):
            f, t = permutation[i: i + 2]
            distance = ((x[t] - x[f]) ** 2 + (y[t] - y[f]) ** 2) ** 0.5
            total_distance += distance
        distances.append(total_distance)
    result = sum(distances) / len(distances)
    print(f"{result:.10f}")
    return


def main():
    def iterate_tokens():
        for line in sys.stdin:
            for word in line.split():
                yield word
    tokens = iterate_tokens()
    N = int(next(tokens))
    x = [int()] * (N)
    y = [int()] * (N)
    for i in range(N):
        x[i] = int(next(tokens))
        y[i] = int(next(tokens))
    solve(N, x, y)


if __name__ == "__main__":
    main()
