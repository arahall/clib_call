import time

import numpy as np
from optimise import lbfgsb
from scipy.optimize import minimize


def bdexp_test(x0, n, model='lbfgsb', debug=False):
    def func(x):
        sum = np.sum([(x[i] + x[i + 1]) * np.exp((x[i] + x[i + 1]) * -x[i + 2])
                                                 for i in range(n - 2)])
        return sum

    def grad(x, g):
        def derivative_wrt_xi():
            """ Derivative of the generalized function with respect to x_i """
            # Term involving x_i, x_{i+1}, and x_{i+2}
            term = (x[i] + x[i + 1])
            exponent = -(x[i] + x[i + 1]) * x[i + 2]
            derivative = (1-x[i + 2] * term ) * np.exp(exponent)
            return derivative

        def derivative_wrt_xi_plus_1():
            """ Derivative of the generalized function with respect to x_{i+1} """
            # Term involving x_i, x_{i+1}, and x_{i+2}
            term = (x[i] + x[i + 1])
            exponent = -(x[i] + x[i + 1]) * x[i + 2]
            derivative = (1-x[i + 2] * term) * np.exp(exponent)
            return derivative

        def derivative_wrt_xi_plus_2():
            """ Derivative of the generalized function with respect to x_{i+2} """
            # Derivative of the exponential term w.r.t x_{i+2}
            term = (x[i] + x[i + 1]) ** 2
            exponent = -(x[i] + x[i + 1]) * x[i + 2]
            derivative = -term * np.exp(exponent)
            return derivative
        for i in range(n):
            g[i] = 0.0
        for i in range(n-2):
            g[i] += derivative_wrt_xi()
            g[i + 1] += derivative_wrt_xi_plus_1()
            g[i + 2] += derivative_wrt_xi_plus_2()


    if model == 'lbfgsb':
        x = np.ones(n, dtype='double') * x0
        lb = -np.ones(n, dtype='double') * np.inf
        ub = np.ones(n, dtype='double') * np.inf
        t1 = time.time()
        result = lbfgsb(func, grad, x, lb, ub, toler=1e-4, debug=debug, max_history=10, eps_factor=1e3)
        t2 = time.time()
        if result is None:
            print("BDEXP Optimization failed")
        else:
            print(f"lbfgsb Optimized x: {x[:10]}: func: {func(x)}")
            print(t2-t1)
            return t2 - t1
    else:
        g = np.empty(n)
        def s_grad(x):
            grad(x,g)
            return g

        bounds = ((-np.inf, np.inf) for _  in range(n))
        x = np.ones(n, dtype='double') * x0

        t1 = time.time()
        result = minimize(func, x, jac=s_grad, method='L-BFGS-B', bounds=bounds)
        t2 = time.time()
        if not result.success:
            print("scipy failed for BDEXP")
        print(f"scipy Optimized x: {result.x[:10]}: func: {func(result.x)}")
        print(t2 - t1)
        return t2 - t1


def beale_test():
    def beale(x_):
        x = x_[0]  # Accessing the first element of the pointer
        y = x_[1]  # Accessing the second element of the pointer
        f = (1.5 - x + x * y) ** 2 + (2.25 - x + x * y * y) ** 2 + (2.625 - x + x * y * y * y) ** 2
        return f

    def beale_g(x_, g):
        x = x_[0]  # Accessing the first element of the pointer
        y = x_[1]  # Accessing the second element of the pointer
        g[0] = (2 * (1.5 - x + x * y) * (-1. + y) + 2 * (2.25 - x + x * y * y) * (-1 + y * y) +
                2 * (2.625 - x + x * y * y * y) * (-1 + y * y * y))
        g[1] = (2 * (1.5 - x + x * y) * x + 2. * (2.25 - x + x * y * y) * 2 * x * y +
                2 * (2.625 - x + x * y * y * y) * 3 * x * y * y)

    x = np.array([0, 0], dtype='double')
    lb = np.array([-4.5, -4.5], dtype='double')
    ub = np.array([4.5, 4.5], dtype='double')

    # Call lbfgsb function to optimize
    result = lbfgsb(beale, beale_g, x, lb, ub)

    if result is not None:
        print(f"Beale Optimized values: {x}, function = {beale(x)}")
    else:
        print("Optimization failed")


def rosenbrock_test(n=2, x0 =-1., model='lbfgsb'):
    def func(x_):
        f = 0
        for i in range(n-1):
            f += (1 - x_[i]) ** 2 + 100 * (x_[i+1] - x_[i] * x_[i]) ** 2
        return f

    def grad(x_, g):
        g[0] = -2 * (1 - x_[0]) - 400 * x_[0] * (x_[1] - x_[0] * x_[0])
        g[n-1] = 200.0 * (x_[n-1] - x_[n-2] * x_[n-2])
        for i in range(1, n-1):
            g[i] = 200 * (x_[i] - x_[i-1]**2) - 400 * x_[i] * (x_[i+1] - x_[i]**2) - 2 *(1-x_[i])

    x = np.ones(n) * x0
    if model == 'lbfgsb':
        lb = np.ones(n) * -np.inf
        ub = np.ones(n) * np.inf

        # Call lbfgsb function to optimize
        t1 = time.time()
        result = lbfgsb(func, grad, x, lb, ub)
        t2 = time.time()

        if result is not None:
            print(f"Rosenbrock Optimized values: {x[:min(n,10)]}, function = {func(x)}")
        else:
            print("Optimization failed")
        return t2 - t1
    else:
        g = np.empty(n)
        def s_grad(x_):
            grad(x_, g)
            return g

        bounds = ((-np.inf, np.inf) for _ in range(n))
        t1 = time.time()
        result = minimize(func, x, jac=s_grad, method='L-BFGS-B', bounds=bounds)
        t2 = time.time()
        if not result.success:
            print("scipy failed for rosenbrock")
        print(f"scipy Optimized x: {result.x[:min(n, 10)]}: func: {func(result.x)}")
        return t2 - t1

def mccormick_test():
    def mccormick(x_):
        x = x_[0]
        y = x_[1]
        return np.sin(x + y) + (x - y) * (x - y) - 1.5 * x + 2.5 * y + 1

    def mccormick_g(x_, g):
        x = x_[0]
        y = x_[1]
        g[0] = np.cos(x + y) + 2 * (x - y) - 1.5
        g[1] = np.cos(x + y) - 2 * (x - y) + 2.5

    x = np.array([0, 0], dtype='double')
    lb = np.array([-1.5, -2], dtype='double')
    ub = np.array([4, 4], dtype='double')

    result = lbfgsb(mccormick, mccormick_g, x, lb, ub)

    if result is not None:
        print(f"Mccormick Optimized values: {x}, function = {mccormick(x)}")
    else:
        print("Optimization failed")


def easom_test():
    def easom(x_):
        x = x_[0]
        y = x_[1]
        h = np.exp(-((x - np.pi) * (x - np.pi) + (y - np.pi) * (y - np.pi)))
        f = -np.cos(x) * np.cos(y) * h
        return f

    def easom_g(x_, g):
        x = x_[0]  # Accessing the first element of the pointer
        y = x_[1]  # Accessing the second element of the pointer
        h = np.exp(-((x - np.pi) * (x - np.pi) + (y - np.pi) * (y - np.pi)))
        g[0] = h * (np.sin(x) * np.cos(y) + np.cos(x) * np.cos(y) * 2 * (x - np.pi))
        g[1] = h * (np.cos(x) * np.sin(y) + np.cos(x) * np.cos(y) * 2 * (y - np.pi))

    x = np.array([4, 4], dtype='double')
    lb = np.array([-100, -100], dtype='double')
    ub = np.array([100, 100], dtype='double')

    result = lbfgsb(easom, easom_g, x, lb, ub)

    if result is not None:
        print(f"Easom Optimized values: {x}, function = {easom(x)}")
    else:
        print("Optimization failed")


def matyas_test():
    def matyas(x_):
        x = x_[0]
        y = x_[1]
        return 0.26 * (x * x + y * y) - 0.48 * x * y

    def matyas_g(x_, g):
        x = x_[0]
        y = x_[1]
        g[0] = 0.52 * x - 0.48 * y
        g[1] = 0.52 * y - 0.48 * x

    x = np.array([-10, -1], dtype='double')
    lb = np.array([-np.inf, -np.inf], dtype='double')
    ub = np.array([np.inf, np.inf], dtype='double')

    result = lbfgsb(matyas, matyas_g, x, lb, ub)

    if result is not None:
        print(f"Matyas Optimized values: {x}, function = {matyas(x)}")
    else:
        print("Optimization failed")


def rastrigin_test(n=2, a=10, x0 = 2, model='lbfgsb'):
    def func(x_):
        f = a * n
        for i in range(n):
            f += x_[i] * x_[i] - 10 * np.cos(2 * np.pi * x_[i])
        return f

    def grad(x_, g):
        for i in range(n):
            g[i] = 2 * x_[i] + 2 * np.pi * a * np.sin(2 * np.pi * x_[i])

    x = np.ones(n, dtype='double') * x0
    if model == 'lbfgsb':
        lb = np.ones(n, dtype='double') * -5.12
        ub = np.ones(n, dtype='double') * 5.12
        t1 = time.time()
        result = lbfgsb(func, grad, x, lb, ub)
        t2 = time.time()

        if result is not None:
            print(f"lbfgs Rastrigin x: {x[:min(10,n)]} function:{func(x)}")
            print(t2 - t1)
            return t2 - t1
        else:
            print("Optimization failed")
    else:
        g = np.empty(n)
        def s_grad(x):
            grad(x,g)
            return g

        bounds = ((-5.12, 5.12) for _  in range(n))

        t1 = time.time()
        result = minimize(func, x, jac=s_grad, method='L-BFGS-B', bounds=bounds)
        t2 = time.time()
        if not result.success:
            print("scipy failed for rastrigin")
        else:
            print(f"scipy Rastrigin x: {result.x[:min(10,n)]} function = {func(result.x)}")
            print(t2 - t1)
            return t2 - t1

def griewank_test(n, x0=1, model='lbfgsb'):
    def func(x_):
        f = 1
        t = 1.
        for i in range(n):
            f += x_[i] * x_[i] / 4000
            t *= np.cos(x_[i] / np.sqrt(i+1))
        f -= t
        return f

    def grad(x_, g):
        t = 1.0
        for i in range(n):
            t *= np.cos(x_[i] / np.sqrt(i+1))
        for i in range(n):
            g[i] = x_[i] / 2000 + t * np.tan(x_[i]/np.sqrt(i+1)) /np.sqrt(i+1)
    x = np.ones(n) * x0
    if model == 'lbfgsb':
        lb = np.ones(n) * -600
        ub = np.ones(n) *  600
        t1 = time.time()
        result = lbfgsb(func, grad, x, lb, ub)
        t2 = time.time()
        if result is not None:
            print(f"Optimized x: {x[:10]}: func: {func(x)}")
        else:
            print("Optimization failed")
        return t2 - t1
    else:
        bounds = ((-600, 600) for _  in range(n))
        g = np.empty(n)
        def s_grad(x_):
            grad(x_, g)
            return g

        t1 = time.time()
        result = minimize(func, x, jac=s_grad, method='L-BFGS-B', bounds=bounds)
        t2 = time.time()
        if not result.success:
            print("scipy failed for Griewank")
        print(f"scipy Optimized x: {result.x[:10]}: func: {func(result.x)}")
        return t2 - t1

def styblinksi_tang_test(n, l, u, x0=1, model='lbfgsb'):
    def func(x_):
        sum = 0.0
        for i in range(n):
            xi = x_[i]
            sum += xi * xi * xi * xi - 16 * xi * xi + 5 * xi;
        return 0.5 * sum

    def grad(x_, g):
        for i in range(n):
            xi = x_[i]
            g[i] = 0.5 * (4 * xi * xi * xi - 32 * xi + 5)

    x = np.ones(n) * x0
    for i in range(1, n, 2):
        x[i] *= -1
    if model == 'lbfgsb':
        lb = np.ones(n) * l
        ub = np.ones(n) *  u
        t1 = time.time()
        result = lbfgsb(func, grad, x, lb, ub)
        t2 = time.time()
        if result is not None:
            print(f"Optimized x: {x[:10]}: func: {func(x)}")
        else:
            print("Optimization failed")
        return t2 - t1
    else:
        bounds = ((l, u) for _  in range(n))
        g = np.empty(n)
        def s_grad(x_):
            grad(x_, g)
            return g

        t1 = time.time()
        result = minimize(func, x, jac=s_grad, method='L-BFGS-B', bounds=bounds)
        t2 = time.time()
        if not result.success:
            print("scipy failed for stylinksi")
        print(f"scipy Optimized x: {result.x[:10]}: func: {func(result.x)}")
        return t2 - t1

# Test the code
def test():  # Load the shared library
    # rosenbrock_test()
    # matyas_test()
    # easom_test()
    # mccormick_test()
    #rastrigin_test(10)
    #
    avg_1 = 0.
    avg_2 = 0.
    n = 1
    for _ in range(n):
        # avg_1 += griewank_test(10, x0=1)
        # avg_2 += griewank_test(10, x0=1, model='scipy')
        avg_1 += rastrigin_test(5000, 10, x0=1)
        avg_2 += rastrigin_test(5000, 10, x0=1, model='scipy')
        # avg_1 += styblinksi_tang_test(5000, -5, 5, 0)
        # avg_2 += styblinksi_tang_test(5000, -5, 5, 0, model='scipy')
        #avg_1 += rosenbrock_test(200, -1)
        #avg_2 += rosenbrock_test(200, -1,'scipy')
    print(avg_1/n, avg_2/n, avg_1/avg_2)

if __name__ == "__main__":
    test()
