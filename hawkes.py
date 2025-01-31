import numpy as np
from scipy.optimize import newton, minimize
from optimise import lbfgsb
import time

# Define the intensity function of the Hawkes process
def intensity(mu, alpha, beta, event_times, t):
    """
    mu: Baseline intensity
    alpha: Excitation parameter
    beta: Decay rate
    event_times: List or array of past event times
    t: Current time at which to evaluate the intensity
    """
    return mu + np.sum(alpha * np.exp(-beta * (t - event_times[event_times < t])))

def compensator_hawkes(mu, alpha, beta, event_times, t_max):
    """
    Calculate the compensator of a Hawkes process with exponential kernel.
    """
    ts_ = event_times[event_times < t_max]
    decay_contributions = 1 - np.exp(-beta * (t_max - ts_))
    comp = mu * t_max + alpha * np.sum(decay_contributions)
    return comp

def log_likelihood(mu, alpha, beta, event_times):
    """
    calculates the loglikelihood given the parameters of a Hawkes process
    :param mu:
    :param alpha:
    :param beta:
    :param event_times:
    :return:
    """
    delta_ts = np.diff(event_times)  # Calculate differences between consecutive event_times
    n = len(event_times)

    # Initialize 'a' and perform recursive computation manually
    a = np.empty(n)
    exp_term = np.exp(-beta * delta_ts)
    a[0] = 0.0
    for i in range(1, n):
        a[i] = exp_term[i - 1] * (1.0 + a[i - 1])

    # Vectorized sum1 and sum2
    delta_t_last = event_times[-1] - event_times
    sum1 = np.sum(np.exp(-beta * delta_t_last) - 1.0)
    sum2 = np.sum(np.log(mu + alpha * a))

    return -mu * event_times[-1] + (alpha / beta) * sum1 + sum2

def log_likelihood_derivs(mu, alpha, beta, event_times):
    """
    calculate the derivatoves of the log likelohood wrt to the parameters
    :param mu:
    :param alpha:
    :param beta:
    :param event_times:
    :return:
    """
    delta_ts = np.diff(event_times)
    n = len(event_times)

    a = np.empty(n)
    b = np.empty(n)
    a[0] = 0.0
    b[0] = 0.0

    exp_term = np.exp(-beta * delta_ts)

    for i in range(1, n):
        a[i] = exp_term[i - 1] * (1.0 + a[i - 1])
        b[i] = -delta_ts[i - 1] * a[i] + exp_term[i - 1] * b[i - 1]

    # Vectorized sum1, sum2, sum3, sum4, and sum5
    delta_t_last = event_times[-1] - event_times
    exp_delta_t_last = np.exp(-beta * delta_t_last)

    sum1 = np.sum(exp_delta_t_last - 1.0)
    sum2 = np.sum(a / (mu + alpha * a))
    sum3 = np.sum(delta_t_last * exp_delta_t_last + (exp_delta_t_last - 1.0) / beta)
    sum4 = np.sum(alpha * b / (mu + alpha * a))
    sum5 = np.sum(1.0 / (mu + alpha * a))

    dl_dmu = -event_times[-1] + sum5
    dl_dalpha = sum1 / beta + sum2
    dl_dbeta = -alpha / beta * sum3 + sum4

    return dl_dmu, dl_dalpha, dl_dbeta

def simulate_hawkes(mu, alpha, beta, T):
    """
    Simulate a Hawkes process using Ogata's thinning algorithm.
    Parameters:
    - mu: Baseline intensity (rate of spontaneous events).
    - alpha: Self-excitation parameter (how much each event increases the rate).
    - beta: Decay rate of self-excitation.
    - T: Time horizon (total time over which the process is simulated).

    Returns:
    - events: List of event times.
    """
    events = []  # To store the event times
    t = 0  # Start time
    lambda_star = mu  # Initialize the maximum intensity

    while t < T:
        # Sample the next time using a Poisson process with intensity lambda_star
        u = np.random.uniform(0, 1)
        t += -np.log(u) / lambda_star

        if t >= T:
            break

        # Calculate the current intensity (intensity just before time t)
        intensity_t = mu + alpha * np.sum(np.exp(-beta * (t - np.array(events))))

        # Accept or reject the event
        u2 = np.random.uniform(0, 1)
        if u2 <= intensity_t / lambda_star:
            events.append(t)  # Accept the event
            lambda_star = intensity_t + alpha  # Update lambda_star
        else:
            lambda_star = intensity_t  # Update lambda_star without event

    return np.array(events)

def simulate_hawkes_ozaki(mu, alpha, beta, u):
    """
    simulates a Hawkes process using Ozaki's thining algorithm
    :param mu:
    :param alpha:
    :param beta:
    :param u:
    :return:
    """
    if np.any(u < 0) or np.any(u > 1):
        raise ValueError("All values in u must be in the range [0, 1].")

    n = len(u)
    event_times = np.empty(n)
    decay_factors = np.empty(n - 1)

    # Initialize the first timestamp using the baseline intensity `mu`
    event_times[0] = -np.log(u[0]) / mu

    # Precompute the decaying exponential term iteratively
    s_values = np.ones(n)  # Array to store s_func values for each event

    for i in range(1, n):
        def func(x):
            delta_t = x - event_times[i - 1]
            exp_decay = 1.0 - np.exp(-beta * delta_t)
            return np.log(u[i]) + mu * delta_t + (alpha / beta) * s_values[i - 1] * exp_decay

        def deriv(x):
            delta_t = x - event_times[i - 1]
            return mu + alpha * s_values[i - 1] * np.exp(-beta * delta_t)

        x0 = event_times[i - 1] - np.log(u[i]) / mu
        event_times[i] = newton(func, x0, fprime=deriv)
        decay_factors[i - 1] = np.exp(-beta * (event_times[i] - event_times[i - 1]))
        s_values[i] = s_values[i - 1] * decay_factors[i - 1] + 1.0

    return event_times

def hawkes_fit(event_times, mu: float = 0.1, alpha: float = .1, beta: float = .1) -> []:
    """
    fits a Hawkes process with parameters mu, alpha beta to a set of event times
    :param event_times:
    :param mu:
    :param alpha:
    :param beta:
    :return:
    """
    def func(x):
        ll = -log_likelihood(x[0], x[1], x[2], event_times)
        return ll

    def grad(x, g):
        derivs = log_likelihood_derivs(x[0], x[1],x[2], event_times)
        g[0] = -derivs[0]
        g[1] = -derivs[1]
        g[2] = -derivs[2]

    x = np.array([mu, alpha, beta], dtype='double')
    lb = np.array([1e-6, 1e-6, 1e-6], dtype='double')
    ub = np.array([np.inf, np.inf, np.inf], dtype='double')

    result = lbfgsb(func, grad, x, lb, ub, toler=1e-2, max_history=15,
                    ln_srch_maxiter=5, debug=False)
    return x if result is not None else None

def hawkes_fit_scipy(event_times, mu: float = 0.1, alpha: float = .1, beta: float = .1) -> []:
    """
    fits a Hawkes process with parameters mu, alpha beta to a set of event times
    :param event_times:
    :param mu:
    :param alpha:
    :param beta:
    :return:
    """
    def func(x):
        ll = -log_likelihood(x[0], x[1], x[2], event_times)
        return ll

    def grad(x):
        derivs = log_likelihood_derivs(x[0], x[1],x[2], event_times)
        return -derivs[0], -derivs[1], -derivs[2]

    x = np.array([mu, alpha, beta], dtype='double')
    bounds = ((1e-6, np.inf), (1e-6, np.inf), (1e-6, np.inf))

    result = minimize(func, x, jac=grad, method='L-BFGS-B', bounds=bounds)
    if result.success:
        return result.x
    return None

# Example usage
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parameters
    mu = .5
    alpha = 2
    beta = 3
    n = 500
    # Example event times
    np.random.seed(12345)
    u = np.random.uniform(0, 1, n)
    event_times = simulate_hawkes_ozaki(mu, alpha, beta, u)
    # event_times = simulate_hawkes(mu, alpha, beta, n)
    print('number of events: ', n)
    for _ in range(10):
    # fit event times to a hawkes process
        t1 = time.time()
        res = hawkes_fit(event_times, .1, .5, .8)
        t2 = time.time()
        print('lbfgs time:', t2-t1)
    print(res)
    for _ in range(10):
        t1 = time.time()
        res = hawkes_fit_scipy(event_times, .1, .5, .8)
        t2 = time.time()
        print('scipy time: ', t2-t1)
    print(res)
    # if res is not None:
    #     # Log-likelihood
    #     print('estimated parameters: ', res)
    #     print('ground truth: ', mu, alpha, beta)
    #     ll = log_likelihood(res[0], res[1], res[2], event_times)
    #     print("Log-likelihood:", ll)
    #     # Derivatives
    #     dmu, dalpha, dbeta = log_likelihood_derivs(res[0], res[1], res[2], event_times)
    #     print("dL/dmu:", dmu)
    #     print("dL/dalpha:", dalpha)
    #     print("dL/dbeta:", dbeta)
    #     t = np.linspace(0, n, num=n, endpoint=True)
    #     intense = np.array([intensity(res[0], res[1], res[2], event_times, event_times[i]) for i in range(n)])
    #     plt.plot(t, intense)
    #     plt.show()
    #     compensator = np.array([compensator_hawkes(res[0], res[1], res[2], event_times, event_times[i]) for i in range(n)])
    #     plt.plot(t, compensator)
    #     plt.plot(t, t)
    #     plt.show()
    # else:
    #     print('fitting failed')
