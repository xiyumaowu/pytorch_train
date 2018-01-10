nums1 = [1,3]
nums2 = []

num = nums1 + nums2
num.sort()

n = len(num)
if n%2 != 0:
    med = num[int((n-1)/2)]/1.0
else:
    med = (num[int((n-2)/2)] + num[int((n+2)/2)-1])/2.0
print(med)