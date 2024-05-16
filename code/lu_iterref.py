import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres
import time

def lu_iterref(A, b, tol_stop = True):
    
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)
    lu, piv = lu_factor(A.astype(np.float32))
    
    r = b.astype(np.float64)
    tol = 10 * np.finfo(np.float64).eps
    relres = []
    rnorm = np.linalg.norm(r)
    bnorm = rnorm
    rnormold = 2*rnorm


    itcount = 0
    while itcount <= 10:
        if tol_stop and rnorm <= tol*bnorm:
            break
        if rnorm >= 0.9*rnormold:
            break

        d = lu_solve((lu.astype(np.float64), piv.astype(np.float64)), r)
        x = x.astype(np.float64) + d.astype(np.float64)
        r = b.astype(np.float64) - A.astype(np.float64).dot(x)
        rnorm = np.linalg.norm(r)
        relres.append((rnorm / bnorm))
        itcount += 1


    return x, relres


if __name__ == '__main__':
    # Example usage
    n = 1000
    xref = np.ones(n)
    A = np.random.rand(n, n)
    b = A @ xref

    # Call the iterative refinement function
    x_iter, residuals = lu_iterref(A, b)

    # Direct solve for comparison
    start_time = time.time()
    x_direct = np.linalg.solve(A, b)
    direct_solve_time = time.time() - start_time

    # Measure the error
    error_func = lambda x: np.linalg.norm(x-xref) / np.linalg.norm(xref)
    error_direct = error_func(x_direct)
    error_iter = error_func(x_iter)

    # Output the results
    print(10 * np.finfo(np.float64).eps)
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    print(f"Error (Direct): {error_direct:.2e}")
    print(f"Error (Iterative): {error_iter:.2e}")
    print(f"Direct solve time: {direct_solve_time:.2e}")
    print(f"Iterations: {len(residuals)}")

