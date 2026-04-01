from typing import List, Tuple, Dict, Any, Optional

Number = float
Matrix = List[List[Number]]
Vector = List[Number]

def _identity(n: int) -> Matrix:
    I = _zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I
def determinant(A: Matrix, eps: float = 1e-12) -> float:
    """
    det(A) via Gaussian elimination with partial pivoting.
    """
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("A must be square for determinant.")

    # copy
    M = _deepcopy_mat(A)
    s = 0
    det_val = 1.0

    for k in range(n):
        # pivot row
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) <= eps:
            return 0.0
        if p != k:
            _swap_rows(M, p, k)
            s += 1

        pivot = M[k][k]
        det_val *= pivot

        # eliminate
        for i in range(k + 1, n):
            if abs(M[i][k]) <= eps:
                continue
            factor = M[i][k] / pivot
            for j in range(k, n):
                M[i][j] -= factor * M[k][j]
            M[i][k] = 0.0

    if s % 2 == 1:
        det_val = -det_val
    return det_val

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

def rref(A: Matrix, eps: float = 1e-12) -> Tuple[Matrix, List[int]]:
    """
    Compute RREF of A (from scratch) with partial pivoting.
    Returns (R, pivot_cols).
    """
    M = _deepcopy_mat(A)
    m = len(M)
    n = len(M[0]) if m > 0 else 0

    r = 0
    pivot_cols: List[int] = []
    for c in range(n):
        p = None
        maxabs = 0.0
        for i in range(r, m):
            val = abs(M[i][c])
            if val > maxabs:
                maxabs = val
                p = i
        if p is None or maxabs <= eps:
            continue
        if p != r:
            _swap_rows(M, p, r)

        # normalize row r
        pivot = M[r][c]
        for j in range(c, n):
            M[r][j] /= pivot

        # eliminate all other rows
        for i in range(m):
            if i == r:
                continue
            factor = M[i][c]
            if abs(factor) <= eps:
                continue
            for j in range(c, n):
                M[i][j] -= factor * M[r][j]
            M[i][c] = 0.0

        pivot_cols.append(c)
        r += 1
        if r == m:
            break

    # clean small numbers
    for i in range(m):
        for j in range(n):
            if abs(M[i][j]) < eps:
                M[i][j] = 0.0

    return M, pivot_cols

def rank_and_basis(A: Matrix, eps: float = 1e-12) -> Dict[str, Any]:
    """
    Returns:
      rank
      pivot_cols
      column_space_basis (from original A columns)
      row_space_basis (non-zero rows of RREF)
      null_space_basis (basis vectors for Ax=0)
    """
    m = len(A)
    n = len(A[0]) if m > 0 else 0

    R, pivot_cols = rref(A, eps=eps)
    rank = len(pivot_cols)

    # Column space basis: pivot columns of original A
    col_basis: List[Vector] = []
    for j in pivot_cols:
        col_basis.append([A[i][j] for i in range(m)])

    # Row space basis: non-zero rows of RREF
    row_basis: List[Vector] = []
    for i in range(m):
        if any(abs(R[i][j]) > eps for j in range(n)):
            row_basis.append(R[i][:])

    # Null space basis: from RREF
    pivot_set = set(pivot_cols)
    free_cols = [j for j in range(n) if j not in pivot_set]
    null_basis: List[Vector] = []

    # For each free variable x_f = 1, others free=0, solve pivot vars = -R[pivot_row, free]
    # Need mapping pivot_col -> pivot_row in R
    pivot_row_for_col: Dict[int, int] = {}
    pr = 0
    for i in range(m):
        # find leading 1
        lead = None
        for j in range(n):
            if abs(R[i][j]) > eps:
                lead = j
                break
        if lead is not None and lead in pivot_set:
            pivot_row_for_col[lead] = i

    for f in free_cols:
        v = [0.0 for _ in range(n)]
        v[f] = 1.0
        for pc in pivot_cols:
            i = pivot_row_for_col[pc]
            # x_pc + sum_{free} R[i][free]*x_free = 0  => x_pc = - R[i][f]
            v[pc] = -R[i][f]
        null_basis.append(v)

    return {
        "rank": rank,
        "pivot_cols": pivot_cols,
        "column_space_basis": col_basis,
        "row_space_basis": row_basis,
        "null_space_basis": null_basis,
        "rref": R,
    }

def verify_solution(A: Matrix, x: Vector, b: Vector, atol: float = 1e-8) -> Dict[str, Any]:
    """
    Verification with NumPy (allowed only for checking).
    Returns residual norm and boolean pass/fail.
    """
    import numpy as np

    A_np = np.array(A, dtype=float)
    x_np = np.array(x, dtype=float)
    b_np = np.array(b, dtype=float)

    r = A_np @ x_np - b_np
    residual_inf = float(np.linalg.norm(r, ord=np.inf))
    ok = residual_inf <= atol
    return {"residual_inf": residual_inf, "ok": ok, "residual": r.tolist()}
