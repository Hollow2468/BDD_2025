import numpy as np

#to find the number of min line cover
def fr(n, L, A):
    f0 = np.where(A[n] == 0)[0]
    diff = [x for x in f0 if x not in L]
    if len(diff) > 0:
        L = np.append(L, diff[0])
        a, b = np.shape(A)
        if n == a - 1:
            return L
        else:
            return fr(n + 1, L, A)
    else:
        for i in range(len(f0)):
            A1 = np.delete(A, f0[i], axis=1)
            nsz = np.array([])
            L1 = fr(0 , nsz, A1)
            a, b = np.shape(A1)
            if len(L1) == a:
                L = np.append(L1, f0[i])
                L = np.where(L >= f0[i], L + 1, L)
                a, b = np.shape(A)
                if n == a - 1:
                    return L
                else:
                    return fr(n + 1, L, A)
        a, b = np.shape(A)
        if n == a - 1:
            L = np.append(L, -1)
            return L
        else:
            L = np.append(L, -1)
            return fr(n + 1, L, A)


A = np.array([[3,7,1],[8,2,5], [9,1,4], [6,3,7], [2,8,6], [5,4,9], [1,7,2]])
c = np.array([3,2,4])

n,k = np.shape(A)
c_sum = np.sum(c)
num = np.arange(1, k+1)
A = np.vstack([num, A])

#copy the hospital
i1 = 0
for i in range(k):
    if c[i] != 1:
        for j in range(c[i]-1):
            ctrlc = A[:, i1]
            A = np.insert(A, i1, ctrlc, axis = 1)
            i1 += 1
    i1 += 1

n1,k1 = np.shape(A)

#to be a square
if n1 < k1:
    zero = np.zeros((k1 - n1 + 1, A.shape[1]), dtype=int)
    A = np.vstack([A, zero])

if n1 > k1:
    zero = np.zeros((A.shape[0], n1 - k1 +1), dtype=int)
    A = np.hstack([A, zero])

index = A[0]
A = np.delete(A, 0, axis = 0)
n2,k2 = np.shape(A)

#create zero
for i in range(n2):
    A[i] = A[i] - np.min(A[i])
for j in range(k2):
    A[:, j] = A[:, j] - np.min(A[:, j])

nsz = np.array([], dtype=int)
L = fr(0, nsz, A)
L = L.astype(int)
cnt = (L == -1).sum()

#adjust the matrix till number of min cover = size of square
while cnt > 0:

    #标记未匹配的行（即 L[i] = -1 的行）
    #从这些未匹配的行开始，沿着 0 元素交替走：
    #从行走到未匹配的 0 → 标记该列
    #从列走到匹配的 0 → 标记该行
    #重复直到没有新的行或列可以标记
    #最小覆盖的结果：
    #未被标记的行 + 被标记的列
    #这一组合就是最少行列覆盖所有 0 的集合。
    h = np.array([], dtype=int)
    l = np.array([], dtype=int)
    nL = np.where(L == -1)[0]
    yL = np.where(L != 1)[0]
    bl = 0
    for i in range(len(nL)):
        while bl == 0:
            1 ==1



    B = np.zeros((n2, n2), dtype=int)

    min_A = np.min(A[A != 0])
    A[B == 0] -= min_A
    A[B == 2] += min_A

    nsz = np.array([], dtype=int)
    L = fr(0, nsz, A)
    L = L.astype(int)
    cnt = (L == -1).sum()

#recover the sequence


print(L)

