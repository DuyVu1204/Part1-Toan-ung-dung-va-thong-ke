from typing import List, Tuple, Dict, Any, Optional

Number = float
Matrix = List[List[Number]]
Vector = List[Number]

def _identity(n: int) -> Matrix:
    I = _zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I

def inverse(A: Matrix, eps: float = 1e-12) -> Matrix:
    """
    Inverse via Gauss-Jordan on [A|I].
    """
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("A must be square for inverse.")

    # build augmented [A | I]
    M = [A[i][:] + _identity(n)[i] for i in range(n)]
    total_cols = 2 * n

    r = 0
    for c in range(n):
        # pivot
        p = max(range(r, n), key=lambda i: abs(M[i][c]))
        if abs(M[p][c]) <= eps:
            raise ValueError("Matrix is singular; inverse does not exist.")
        if p != r:
            _swap_rows(M, p, r)

        # normalize pivot row
        pivot = M[r][c]
        for j in range(c, total_cols):
            M[r][j] /= pivot

        # eliminate other rows
        for i in range(n):
            if i == r:
                continue
            if abs(M[i][c]) <= eps:
                continue
            factor = M[i][c]
            for j in range(c, total_cols):
                M[i][j] -= factor * M[r][j]
            M[i][c] = 0.0
        r += 1
        if r == n:
            break

    invA = [row[n:] for row in M]
    return invA