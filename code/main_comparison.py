import lu_iterref, gmres_iterref
import numpy as np
from numpy import linalg
import time
import matplotlib.pyplot as plt


def evaluate(A, b, cond):
    # GMRES based iterative refinement function
    gmres_start_time = time.time()
    x_gmres_iter, gmres_residuals = gmres_iterref.gmres_iterref(A, b, tol_stop= True)
    gmres_solve_time = time.time() - gmres_start_time

    # LU based iterative refinement function
    lu_start_time = time.time()
    x_lu_iter, lu_residuals = lu_iterref.lu_iterref(A, b, tol_stop= True)
    lu_solve_time = time.time() - lu_start_time

    xref = np.ones(A.shape[0])

    # Direct solve for comparison
    direct_start_time = time.time()
    x_direct = np.linalg.solve(A, b)
    direct_solve_time = time.time() - direct_start_time

    # Measure the error
    error_func = lambda x: np.linalg.norm(x-xref) / np.linalg.norm(xref)
    error_direct = error_func(x_direct)
    error_gmres_iter = error_func(x_gmres_iter)
    error_lu_iter = error_func(x_lu_iter)

    # Output the results
    print(f"Matrix size: {A.shape[0]}x{A.shape[0]}")
    print(f"Condition number: {cond:.2e}")
    print(f"Error (Direct): {error_direct:.2e}")
    print(f"Error (GMRES-based IR): {error_gmres_iter:.2e}")
    print(f"Error (LU-based IR): {error_lu_iter:.2e}")
    print(f"Direct solve time: {direct_solve_time:.2e}")
    print(f"GMRES IR solve time: {gmres_solve_time:.2e}")
    print(f"LU IR solve time: {lu_solve_time:.2e}")
    print(f"Iterations (GMRES-based IR): {len(gmres_residuals)}")
    print(f"Iterations (LU-based IR): {len(lu_residuals)}")


    result = (error_direct, error_gmres_iter, error_lu_iter, direct_solve_time,
               gmres_solve_time, lu_solve_time, len(gmres_residuals), len(lu_residuals))

    return result



if __name__ == '__main__':

    # Parameters
    size_list = [50, 500, 1000, 2000, 5000, 8000, 12000, 15000]  #matrix size
    size_results = {size: [] for size in size_list}
    cond_results = {}

    # evaluate matrix with different size
    for n in size_list:
        xref = np.ones(n)
        A = np.random.rand(n, n)
        b = A @ xref
        cond = linalg.cond(A)

        result_size = evaluate(A, b, cond)

        size_results[n].append(result_size)


    # evaluate matrix with different condition number in same size
    n = 10000
    for i in range(20):
        xref = np.ones(n)
        A = np.random.rand(n, n)
        b = A @ xref
        cond = linalg.cond(A)

        result_cond = evaluate(A, b, cond)

        if cond not in cond_results:
            cond_results[cond] = []
        cond_results[cond].append(result_cond)



#plotting the result
# prepare data for plot

import matplotlib.pyplot as plt

# plot1：avg acc（size）
sizes = sorted(size_results.keys())
gmres_acc_by_size = [np.mean([r[1] for r in size_results[size]]) for size in sizes]
lu_acc_by_size = [np.mean([r[2] for r in size_results[size]]) for size in sizes]

plt.figure(figsize=(10, 6))
plt.semilogy(sizes, gmres_acc_by_size, 'o-', label='GMRES')
plt.semilogy(sizes, lu_acc_by_size, 'o--', label='LU')
plt.title('Average Accuracy (Error) by Size')
plt.xlabel('Matrix Size')
plt.ylabel('Average Error')
plt.legend()
plt.show()

# plot2：avg time（size）
gmres_time_by_size = [np.mean([r[4] for r in size_results[size]]) for size in sizes]
lu_time_by_size = [np.mean([r[5] for r in size_results[size]]) for size in sizes]

plt.figure(figsize=(10, 6))
plt.semilogy(sizes, gmres_time_by_size, 'o-', label='GMRES')
plt.semilogy(sizes, lu_time_by_size, 'o--', label='LU')
plt.title('Average Computational Time by Size')
plt.xlabel('Matrix Size')
plt.ylabel('Average Time (seconds)')
plt.legend()
plt.show()

# plot3：error（cond）
conds = sorted(cond_results.keys())
gmres_acc_by_cond = [np.mean([r[1] for r in cond_results[cond]]) for cond in conds]
lu_acc_by_cond = [np.mean([r[2] for r in cond_results[cond]]) for cond in conds]

plt.figure(figsize=(10, 6))
plt.semilogy(conds, gmres_acc_by_cond, 'o-', label='GMRES')
plt.semilogy(conds, lu_acc_by_cond, 'o--', label='LU')
plt.title('Accuracy (Error) by Condition Number')
plt.xlabel('Condition Number (log scale)')
plt.xscale('log')
plt.ylabel('Error')
plt.legend()
plt.show()

# plot4：error time（cond）
gmres_time_by_cond = [np.mean([r[4] for r in cond_results[cond]]) for cond in conds]
lu_time_by_cond = [np.mean([r[5] for r in cond_results[cond]]) for cond in conds]

plt.figure(figsize=(10, 6))
plt.semilogy(conds, gmres_time_by_cond, 'o-', label='GMRES')
plt.semilogy(conds, lu_time_by_cond, 'o--', label='LU')
plt.title('Computational Time by Condition Number')
plt.xlabel('Condition Number (log scale)')
plt.xscale('log')
plt.ylabel('Time (seconds)')
plt.legend()
plt.show()

error_difference_size = [gm - lu for gm, lu in zip(gmres_acc_by_size, lu_acc_by_size)]

# plot4: error difference
plt.figure(figsize=(10, 6))
plt.semilogy(sizes, error_difference_size, 'o-', label='Error Difference (GMRES - LU)')
plt.xlabel('Matrix Size')
plt.ylabel('Error Difference')
plt.title('Error Difference Between GMRES and LU Across Matrix Sizes')
plt.legend()
plt.grid(True)
plt.show()

error_difference_conds = [gm - lu for gm, lu in zip(gmres_acc_by_cond, lu_acc_by_cond)]

# plot 5: error difference
plt.figure(figsize=(10, 6))
plt.semilogy(conds, error_difference_conds, 'o-', label='Error Difference (GMRES - LU)')
plt.xlabel('Condition Number (log scale)')
plt.xscale('log')
plt.ylabel('Error Difference')
plt.title('Error Difference Between GMRES and LU Across Matrix Condition Number')
plt.legend()
plt.grid(True)
plt.show()


