from scipy.special import betaln
from scipy.stats import norm
from math import exp
from numpy import log, mean, var, cov, sqrt, NaN, array, nanmax
import numpy as np


def run_bayesian_binomial(alpha_a, beta_a, alpha_b, beta_b, conf=0.95):
    total_prob = 0.0
    for i in range(alpha_b - 1):
        total_prob += exp(
            betaln(alpha_a + i, beta_b + beta_a) - log(beta_b + i) - betaln(1 + i, beta_b) - betaln(alpha_a, beta_a))
    if total_prob > conf:
        return True, total_prob
    else:
        return False, total_prob
    return False, 0


def calculate_tau(alpha, sigma, truncation):
    if not type(alpha) == float or alpha >= 1 or alpha <= 0:
        raise TypeError("Alpha needs to be between 0 and 1")
    b = (2 * log(pow(alpha, -1))) / pow((truncation * pow(sigma, 2)), 1 / 2)
    return round(pow(sigma, 2) * (norm.cdf(-b) / ((1 / b) * norm.pdf(b) - norm.cdf(-b))), 2)


def run_msprt(x, y, alpha, tau, theta, distribution, xpre=None, ypre=None, burn_in=1000):
    n = min(len(x),len(y))
    x = x[0:n]
    y = y[0:n]
    z = x - y
    out = np.empty((n))
    out.fill(np.NaN)
    if distribution == "bernoulli":
        if xpre is not None and ypre is not None:
            for i in range(burn_in, len(x)):
                k = 0.5 * ((cov(xpre[1:i], x[1:i]) / var(xpre[1:i])) + (cov(ypre[1:i], y[1:i]) / var(ypre[1:i])))
                xn = x[1:i] - k * xpre[1:i]
                yn = y[1:i] - k * ypre[1:i]
                Vn = mean(xn[1:i]) * (1 - mean(xn[1:i])) + mean(yn[1:i]) * (1 - mean(yn[1:i]))
                out[i] = sqrt(Vn / (Vn + i * pow(tau, 2))) * exp(
                    (pow(i, 2) * pow(tau, 2) * pow(mean(xn[1:i]) - mean(yn[1:i]) - theta, 2)) /
                    (2 * Vn * (Vn + i * pow(tau, 2))))
        elif xpre is None or ypre is None:
            for i in range(burn_in, len(z)):
                Vn = mean(x[0:i]) * (1 - mean(x[0:i])) + mean(y[0:i]) * (1 - mean(y[0:i]))
                out[i] = sqrt(Vn / (Vn + i * pow(tau, 2))) * exp(
                    (pow(i, 2) * pow(tau, 2) * pow(mean(z[1:i]) - theta, 2)) / (2 * Vn * (Vn + i * pow(tau, 2))))
        out[1:burn_in] = 0

    if nanmax(out) > pow(alpha, -1):
        num_rejections = np.min(np.where(out > pow(alpha, -1)))
    else:
        num_rejections = len(z)

    sig = True if num_rejections < len(x) else False
    return sig, num_rejections


if __name__ == "__main__":
    import pandas as pd
    tester_exp = pd.read_csv('/Users/alan/Documents/mas/mas-thesis/test_data/exp_0.csv')
    x = tester_exp[tester_exp.group == 'control']['convs'].to_numpy()
    y = tester_exp[tester_exp.group != 'control']['convs'].to_numpy()
    min_n = min(len(x), len(y))
    t = calculate_tau(0.05, .24, 10000)
    m_result = run_msprt(x[0:min_n], y[0:min_n], tau=t, theta=0, distribution="bernoulli", alpha=0.05)
    print(m_result)
