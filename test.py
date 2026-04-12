import sys

n = int(input())
data = []
for _ in range(n):
    l, r = map(int, sys.stdin.readline().split())
    data.append((l, "+"))
    data.append((r, "-"))

res = 0
curr = 0
start_elem = 0
data.sort()
for i in range(n * 2):
    if data[i][1] == "+":
        if curr == 0:
            start_elem = data[i][0]
        curr += 1
    else:
        curr -= 1
        if curr == 0:
            res += data[i][0] - start_elem
print(res)
