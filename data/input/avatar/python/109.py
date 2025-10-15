H, W, N = map(int, input().split())
s_rr, s_cc = map(int, input().split())
S = input()
T = input()
Judge = False
S_L, S_R, S_U, S_D = 0, 0, 0, 0
T_L, T_R, T_U, T_D = 0, 0, 0, 0
for x in range(N):
    if S[x] == 'L':
        S_L += 1
        if S_L - T_R - s_cc == 0:
            Judge = True
    elif S[x] == 'R':
        S_R += 1
        if s_cc + (S_R - T_L) == W + 1:
            Judge = True
    elif S[x] == 'U':
        S_U += 1
        if S_U - T_D - s_rr == 0:
            Judge = True
    elif S[x] == 'D':
        S_D += 1
        if s_rr + (S_D - T_U) == H + 1:
            Judge = True
    if T[x] == 'L':
        if S_R - T_L + s_cc != 1:
            T_L += 1
    if T[x] == 'R':
        if s_cc + (T_R - S_L) != W:
            T_R += 1
    if T[x] == 'U':
        if S_D - T_U + s_rr != 1:
            T_U += 1
    if T[x] == 'D':
        if s_rr + (T_D - S_U) != H:
            T_D += 1
if Judge:
    print('NO')
else:
    print('YES')
