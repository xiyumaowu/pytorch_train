s = "bb"

import time
n = len(s)
start = time.time()
ans=0; i=0; j=1
a1 = 0; a2 = 0
for i in range(0, n):
    for j in range(i+1, n):
        if s[j] == s[i]:
            if ans < j-i:
                a1 = i
                a2 = j
            ans = max(ans, j-i)
print(a1, a2)
print(s[a1:a2+1])
end = time.time()
print("spend time: ", end-start)