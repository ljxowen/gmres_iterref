import numpy as np
import lu_iterref, gmres_iterref
import matplotlib.pyplot as plt
from scipy.linalg import hilbert
import time

if __name__ == '__main__':
    n = 1000

    U, _ = np.linalg.qr(np.random.rand(n, n))
    V, _ = np.linalg.qr(np.random.rand(n, n))

    # create sigular values for matrix
    singular_values = np.logspace(0, -np.log10(1e8), n)

    # consturct matrix A = U * S * V.T
    S = np.diag(singular_values)
    A = U @ S @ V.T 
    cond = np.linalg.cond(A)

    # check the condition number and if it's non-symmetric
    sym_bool = not np.array_equal(A, A.T)
    print("Matrix A is non-symmetric:" if sym_bool else "Matrix A is symmetric.")
    print(f"Condition number of A: {cond:.2e}")   

    xref = np.ones(n)
    b = A @ xref

    start_time = time.time()
    x_iter, residuals = gmres_iterref.gmres_iterref(A, b, tol_stop= True)
    solve_time = time.time() - start_time

    lu_start_time = time.time()
    lu_x_iter, lu_residuals = lu_iterref.lu_iterref(A, b, tol_stop= True)
    lu_solve_time = time.time() - lu_start_time

    # Measure the error
    error_func = lambda x: np.linalg.norm(x-xref) / np.linalg.norm(xref)
    error = error_func(x_iter)
    lu_error = error_func(lu_x_iter)

    # Output the results
    print(f"Matrix size: {A.shape[0]}x{A.shape[0]}")
    print(f"GMRES Error: {error:.2e}")
    print(f"GMRES IR solve time: {solve_time:.2e}")
    print(f"Iterations (GMRES-based IR): {len(residuals)}") 

    print(f"LU Error: {lu_error:.2e}")
    print(f"LU IR solve time: {lu_solve_time:.2e}")
    print(f"Iterations (LU-based IR): {len(lu_residuals)}") 


    plt.figure(figsize=(10, 6))
    x1 = np.arange(1, len(residuals) + 1)
    x2 = np.arange(1, len(lu_residuals) + 1)
    plt.semilogy(x1, residuals, marker = 'o', linestyle = '-', linewidth = 3, markersize = 8, label='GMRES')
    plt.semilogy(x2, lu_residuals, marker = 'o', linestyle = '-', linewidth = 3, markersize = 8, label='LU')
    plt.title("Error Convergence Over Iterations")
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.legend()
    #plt.yscale('log')
    plt.show()
