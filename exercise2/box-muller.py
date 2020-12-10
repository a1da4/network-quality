import numpy as np
import matplotlib.pyplot as plt


def box_muller(mu_1, sigma_1, mu_2, sigma_2, num=1000):
    """Box-Muller methds
    :param mu_1: average of normal distribution 1
    :param sigma_1: standard deviation of normal distribution 1
    :param mu_2: average of normal distribution 2
    :param sigma_2: standard deviation of normal distribution 2
    :param num: number of samples
    """
    X_1 = []
    X_2 = []
    obtained = 0
    while obtained < num:
        x_1 = np.random.normal(mu_1, sigma_1)
        x_2 = np.random.normal(mu_2, sigma_2)
        print("sample x_1 and x_2")
        print(f" - x1: {x_1}, x2: {x_2}")
        w = x_1 ** 2 + x_2 ** 2
        print(f" - w: {w}")
        if w > 1:
            continue
        else:
            y = np.sqrt(-2 * np.log(w) / w)
            x_1 *= y
            x_2 *= y
            X_1.append(x_1)
            X_2.append(x_2)
            obtained += 1
    return X_1, X_2


def plot_scatter(X_1, X_2):
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(X_1, X_2)
    plt.show()


if __name__ == "__main__":
    np.random.seed(1)
    mu_1 = 0
    sigma_1 = 1
    mu_2 = 0
    sigma_2 = 1
    X_1, X_2 = box_muller(mu_1, sigma_1, mu_2, sigma_2)
    plot_scatter(X_1, X_2)