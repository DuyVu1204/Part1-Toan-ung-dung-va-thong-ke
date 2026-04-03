def gaussian_eliminate(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    s = 0
    eps = 1e-10
    for k in range(n):
        p = k 
        for i in range(k + 1, n):
            if abs(M[i][k]) > abs(M[p][k]) :
                p = i
        if abs(M[p][k]) < eps:
            raise ValueError("Ma trận suy biến")
        if p != k:
            M[k], M[p] = M[p], M[k]
            s += 1
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]
        U = [row[:n] for row in M]
        c = [row[n] for row in M]
        x = back_substitution(U, c)
    return M, x, s
    
def back_substitution(U, c):
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (U[i][-1] - s) / U[i][i]
    return x

def determinant(A):
    n = len(A)
    M = [row[:] for row in A]
    s = 0
    eps = 1e-10
    for k in range(n):
        p = k 
        for i in range(k + 1, n):
            if abs(M[i][k]) > abs(M[p][k]) :
                p = i
        if abs(M[p][k]) < eps:
            return 0.0
        if p != k:
            M[k], M[p] = M[p], M[k] 
            s += 1
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n):
                M[i][j] -= factor * M[k][j]

    det = 1.0
    for i in range(n):
        det *= M[i][i]
    return det * (-1) ** s