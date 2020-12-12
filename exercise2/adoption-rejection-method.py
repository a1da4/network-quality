# 採択棄却法で f(x)=0.9281 * exp(-(x-4)^2) + 0.1856 - 0.0742 * (x-1)^2 に従う乱数を生成
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """target probability density function"""
    return 0.9281 * np.exp(-((x - 4) ** 2)) + 0.1856 - 0.0742 * (x - 1) ** 2


def N(x, sigma, mu):
    """ gaussian probability density function"""
    return (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma))


def r(x, sigma_1, mu_1, lambda_1, sigma_2, mu_2):
    """pseudo probability density function: gaussian mixture
    :param lambda_1: weight of gaissian_1
    """
    return lambda_1 * N(x, sigma_1, mu_1) + (1 - lambda_1) * N(x, sigma_2, mu_2)


def rejection(num, x, c, sigma_1, mu_1, lambda_1, sigma_2, mu_2):
    """rejection method
    :param num: number of samples
    """
    samples = []
    i = 0
    while len(samples) < num:
        # for i in range(10):
        i += 1
        if np.random.rand() <= lambda_1:
            y = np.random.normal(loc=mu_1, scale=np.sqrt(sigma_1))
        else:
            y = np.random.normal(loc=mu_2, scale=np.sqrt(sigma_2))
        u = np.random.rand()
        threshold = f(y) / (c * r(y, sigma_1, mu_1, lambda_1, sigma_2, mu_2))
        if u <= threshold:
            samples.append(y)

    return samples


if __name__ == "__main__":
    np.random.seed(1)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.xlabel("x")
    plt.ylabel("f(x)")

    # f(x)
    x = np.arange(0, 4.51, 0.01)
    fx = f(x)
    ax1.plot(x, fx, label="f(x)")

    # cr(x)
    sigma_1 = 0.5
    mu_1 = 4
    lambda_1 = 0.5
    sigma_2 = 1
    mu_2 = 1
    c = 2
    rx = r(x, sigma_1, mu_1, lambda_1, sigma_2, mu_2)
    crx = rx * c
    ax1.plot(x, crx, label="cr(x)")

    # 採択棄却法による乱数生成
    num = 1000
    samples = rejection(num, x, c, sigma_1, mu_1, lambda_1, sigma_2, mu_2)
    ax2.hist(samples, bins=20, label="sampled")

    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()

    ax1.legend(handler1 + handler2, label1 + label2)
    plt.savefig("random-numbers_from_adoption-rejection.png")
