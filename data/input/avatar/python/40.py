import sys
sys.setrecursionlimit(10 ** 5)


class Node:
    def __init__(self):
        self.val = None
        self.color = None
        self.a = []


n = int(input())
nodes = []
for i in range(n):
    nodes.append(Node())
for i in range(n - 1):
    u, v, w = map(int, input().split())
    node1 = nodes[u - 1]
    node2 = nodes[v - 1]
    node1.val = u
    node2.val = v
    node1.a.append([node2, w])
    node2.a.append([node1, w])
root = nodes[0]
root.color = False
nodeSet = set()


def traverse(node, distance):
    if node in nodeSet:
        return
    else:
        nodeSet.add(node)
    for pair in node.a:
        adjNode = pair[0]
        dis = pair[1]
        if (distance + dis) % 2 == 0:
            adjNode.color = root.color
        else:
            adjNode.color = not root.color
        traverse(adjNode, distance + dis)


traverse(root, 0)
for i in range(n):
    node = nodes[i]
    if node.color:
        print(1)
    else:
        print(0)
