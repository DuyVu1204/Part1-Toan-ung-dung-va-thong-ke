def gaussian_eliminate(A, b):
    # Tạo ma trận M = [A | b]
    n = len(A)
    m = len(A[0]) if A else 0
    M = [A[i][:] + [b[i]] for i in range(n)]
    s = 0
    eps = 1e-12
    pivot_cols = []
    pivot_rows = []

    # Thực hiện loại bỏ Gaussian
    for k in range(n):
        # Tìm dòng pivot
        pivot_row = -1
        max_val = 0
        for i in range(k, n):
            if abs(M[i][k]) > max_val:
                max_val = abs(M[i][k])
                pivot_row = i
        if max_val < eps:
            continue

        # Hoán đổi 2 dòng (nếu cần)
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
            s += 1

        pivot_cols.append(k)
        pivot_rows.append(k)

        # Loại bỏ các dòng dưới dòng pivot
        for i in range(k + 1, n):
            if abs(M[k][k]) < eps:
                continue
            factor = M[i][k] / M[k][k]
            for j in range(k, m + 1):
                M[i][j] -= factor * M[k][j]

    # Kiểm tra nghiệm 
    free_vars = None
    x = None
    for i in range(n):
        all_zero = True
        for j in range(m):
            if abs(M[i][j]) > eps:
                all_zero = False
                break
        if all_zero and abs(M[i][-1]) > eps:
            return M, None, s, None, None   # Hệ vô nghiệm
        
    # Xác định biến tự do
    if len(pivot_cols) < m:
        free_vars = [j for j in range(m) if j not in pivot_cols]
        for i in range(len(pivot_rows) - 1, -1, -1):
            row = pivot_rows[i]
            pivot = pivot_cols[i]
            if abs(M[row][pivot]) > eps:
                factor = M[row][pivot]
                for j in range(m + 1):
                    M[row][j] /= factor
            for r in range(row):
                if abs(M[r][pivot]) > eps:
                    factor = M[r][pivot]
                    for j in range(m + 1):
                        M[r][j] -= factor * M[row][j]
        return M, None, s, pivot_cols, free_vars

    # Trích xuất U và c từ M
    U = []
    c = []
    for i in range(n):
        if any(abs(M[i][j]) > eps for j in range(m)):
            U.append(M[i][:m])
            c.append(M[i][m])

    if len(U) == m: 
        x = back_substitution(U, c)

    return M, x, s, pivot_cols, free_vars
    
def back_substitution(U, c):
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range (i + 1, n):
            s += U[i][j] * x[j]
        if abs(U[i][i]) < 1e-12:
                raise ValueError("Hệ vô nghiệm")
        x[i] = (c[i] - s) / U[i][i]
    return x

def determinant(A):
    n = len(A)
    M = [row[:] for row in A]
    det_sign = 1
    eps = 1e-12
    for k in range(n):
        pivot_row = k 
        max_val = abs(M[k][k])
        for i in range(k + 1, n):
            if abs(M[i][k]) > max_val :
                max_val = abs(M[i][k])
                pivot_row = i
        if max_val < eps:
            return 0.0
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k] 
            det_sign *= -1
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n):
                M[i][j] -= factor * M[k][j]

    det = det_sign
    for i in range(n):
        det *= M[i][i]
    return det

def general_solution(A, b):
    eps = 1e-12
    M, x, s, pivot_cols, free_vars = gaussian_eliminate(A, b)
    if free_vars is None:
        if x is None:
            print("\nHệ vô nghiệm")
        else:
            print("\nHệ có nghiệm duy nhất: x = ", x)
        return None, None, None
    
    n = len(A)
    m = len(A[0])
    pivot_rows = []
    for i in range(n):
        for j in pivot_cols:
            if abs(M[i][j]) > eps:
                pivot_rows.append(i)
                break
    
    # Xây dựng nghiệm riêng
    particular = [0.0] * m
    for idx, pc in enumerate(pivot_cols):
        row = pivot_rows[idx]
        particular[pc] = M[row][m]

    # Xây dựng cơ sở không gian nghiệm
    nullspace = []
    for free_idx, free_var in enumerate(free_vars):
        basis_vector = [0.0] * m
        basis_vector[free_var] = 1.0
        for idx, pc in enumerate(pivot_cols):
            row = pivot_rows[idx]
            coeff = M[row][free_var]
            if abs(coeff) > eps:
                basis_vector[pc] = -coeff
            else:
                basis_vector[pc] = 0.0
        nullspace.append(basis_vector)

    return particular, nullspace, free_vars

def print_general_solution(A, b):
    particular, nullspace, free_vars = general_solution(A, b)
    if particular is None:
        return
    
    m = len(A[0])
    
    print("CÔNG THỨC NGHIỆM TỔNG QUÁT")
    
    print("\nNghiệm riêng (đặt các biến tự do = 0):")
    for i in range(m):
        print(f"  x_{i+1} = {particular[i]:.4f}")
    
    if nullspace and len(nullspace) > 0:
        print("\nNghiệm tổng quát:")
        print(f"  x = x_p + c1*v1 + c2*v2 + ... + c{len(A)}*v{len(A)}")
        print("\n  Trong đó:")
        
        for idx, basis in enumerate(nullspace):
            print(f"\n  v{idx+1} (ứng với biến tự do x_{free_vars[idx]+1}):")
            for i in range(m):
                val = basis[i]
                if abs(val) > 1e-12:
                    if i == free_vars[idx]:
                        print(f"    x_{i+1} = 1")
                    elif abs(val - 1) < 1e-12:
                        print(f"    x_{i+1} = 1")
                    elif abs(val + 1) < 1e-12:
                        print(f"    x_{i+1} = -1")
                    else:
                        print(f"    x_{i+1} = {val:.4f}")
        
        print(f"\n  Các biến tự do: ", end="")
        for fv in free_vars:
            print(f"x_{fv+1} ", end="")
        print()
    else:
        print("\n(Đây là nghiệm duy nhất, không có biến tự do)")

if __name__ == "__main__":
    # Ví dụ 1: Hệ có vô số nghiệm
    print("\n" + "="*60)
    print("VÍ DỤ 1: Hệ vô số nghiệm")
    
    A1 = [
        [1, 2, 3],
        [2, 4, 6],
        [1, 1, 1]
    ]
    b1 = [4, 8, 3]
    
    print("\nHệ phương trình:")
    for i in range(len(A1)):
        eq = " + ".join([f"{A1[i][j]}x_{j+1}" for j in range(len(A1[0]))])
        print(f"  {eq} = {b1[i]}")
    
    print_general_solution(A1, b1)
    
    # Ví dụ 2: Hệ vô số nghiệm khác
    print("\n" + "="*60)
    print("VÍ DỤ 2: Hệ vô số nghiệm (3 ẩn, 2 phương trình)")
    
    A2 = [
        [1, 2, -1],
        [2, 4, -2]
    ]
    b2 = [3, 6]
    
    print("\nHệ phương trình:")
    for i in range(len(A2)):
        eq = " + ".join([f"{A2[i][j]}x_{j+1}" for j in range(len(A2[0]))])
        print(f"  {eq} = {b2[i]}")
    
    print_general_solution(A2, b2)
    
    # Ví dụ 3: Hệ có nghiệm duy nhất
    print("\n" + "="*60)
    print("VÍ DỤ 3: Hệ nghiệm duy nhất")
    
    A3 = [
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ]
    b3 = [8, -11, -3]
    
    print("\nHệ phương trình:")
    for i in range(len(A3)):
        eq = " + ".join([f"{A3[i][j]}x_{j+1}" for j in range(len(A3[0]))])
        print(f"  {eq} = {b3[i]}")
    
    print_general_solution(A3, b3)

    print("\n" + "="*60)
    print("VÍ DỤ 4: Kiểm tra định thức")

    test_cases = [
        ([[1, 2], [3, 4]], -2),           # det = -2
        ([[2, 1], [1, 2]], 3),            # det = 3
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 0),  # det = 0
        ([[2, 0, 0], [0, 3, 0], [0, 0, 5]], 30), # det = 30
        ([[1, 1, 0], [1, 0, 1], [0, 1, 1]], -2)  # det = -2
    ]

    for i, (A, expected) in enumerate(test_cases, 1):
        result = determinant(A)
        status = "✓" if abs(result - expected) < 1e-12 else "✗"
        print(f"Test {i}: det = {result:.4f} (expected {expected}) {status}")