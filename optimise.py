import ctypes as ct
import time

try:
    FUNC = ct.WINFUNCTYPE(ct.c_double, ct.POINTER(ct.c_double))  # For func (returns double, takes double*)
    GRAD = ct.WINFUNCTYPE(None, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))
    libname = "python_call.dll"
    path = "C:\\Users\\User\\source\\repos\\python_call\\x64\\Release\\"
    lib = ct.windll.LoadLibrary(path + libname)
    # Define the argument types for the lbfgsb function
    lib.lbfgsb.argtypes = [FUNC, GRAD, ct.c_size_t,
                           ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                           ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
                           ct.c_double, ct.c_bool]

    # Define the return type for the lbfgsb function
    lib.lbfgsb.restype = ct.c_bool
except OSError as e:
    raise RuntimeError(f"Failed to load the shared library: {e}")


def lbfgsb(func, grad, x, lb, ub,
           max_iter=100,
           ln_srch_maxiter=10,
           max_history=5,
           toler=0.0001,
           c1=1e-4, c2=0.9, max_alpha=2.5, eps_factor=1e7, debug=False):
    """
    A wrapper for the LBFGSB optimization routine from a shared library.

    Parameters:
        func (callable): The objective function to minimize.
        grad (callable): The gradient of the objective function.
        x (list of float): Initial guess for the variables.
        lb (list of float): Lower bounds for the variables.
        ub (list of float): Upper bounds for the variables.
        max_iter (int): Maximum number of iterations (default: 100).
        ln_srch_maxiter: maximum number of line search iterations (default=10)
        max_history (int): Number of history steps to keep (default: 5).
        toler (float): Tolerance for convergence (default: 1e-4).
        c1 (float): Wolfe condition constant (default: 1e-4).
        c2 (float): Wolfe condition constant (default: 0.9).
        max_alpha (float): Maximum alpha for line search (default: 2.5).
        debug (bool): Enable debug mode (default: False).

    Returns:
        list of float or None: Optimized variables, or None if optimization failed.
    """

    n = len(x)
    if n != len(lb) or n != len(ub):
        raise ValueError("Dimensions of 'x', 'lb', and 'ub' must match.")

    # Wrap the function and gradient callbacks
    f = FUNC(func)
    g = GRAD(grad) if grad is not None else GRAD()

    # Create ctypes arrays for x, lb, and ub
    DbleArray = ct.c_double * n
    xa = DbleArray(*x)
    lba = DbleArray(*lb)
    uba = DbleArray(*ub)

    # Call the LBFGSB function from the shared library
    res = lib.lbfgsb(f, g, n, xa, lba, uba, max_history, max_iter, ln_srch_maxiter,
                     ct.c_double(toler), ct.c_double(c1), ct.c_double(c2),
                     ct.c_double(max_alpha), ct.c_double(eps_factor), ct.c_bool(debug))
    # Check result and update the input array x if successful
    if not res:
        return None

    x[:] = xa
    return x
