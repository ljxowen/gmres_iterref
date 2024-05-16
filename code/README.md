1. To perform section 3.1 result:
In main_comparison.py
set tol_stop = True in line 11 and 16
and run

2. to perfom section 3.2 result:
In main_comparison.py
set tol_stop = False in line 11 and 16
and run

3. to perform section 3.3 result:
run main_ill_cond.py
Note: if receve a error, this is due the code is trying to generate a ill-conditioned matrix which may cause the lu_factor() unable to handle the sigular value, just run it again until it's work.