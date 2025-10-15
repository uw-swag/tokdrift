s = input()
res = []
punctuation = [',', '.', '!', '?']
for i in range(len(s)):
    if i >= 1:
        if s[i] == ' ':
            if res[- 1] != ' ':
                res.append(s[i])
            else:
                continue
        else:
            if s[i] in punctuation:
                if res[- 1] == ' ':
                    res.pop()
                res.append(s[i])
                res.append(' ')
            else:
                res.append(s[i])
    else:
        if s[i] == ' ':
            continue
        if s[i] in punctuation:
            continue
        else:
            res.append(s[i])
print(''.join(res))
