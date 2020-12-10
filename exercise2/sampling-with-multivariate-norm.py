import numpy as np
import matplotlib.pyplot as plt


def plot_scatter(Y_1, Y_2):
    plt.xlabel("Y1")
    plt.ylabel("Y2")
    plt.scatter(Y_1, Y_2)
    plt.show()


if __name__ == "__main__":
    np.random.seed(1)
    mu_1 = 100
    mu_2 = 200
    N = 1000
    mean = np.array([mu_1, mu_2])
    cov = np.array(
        [
            [(0.3 * mu_1) ** 2, 0.7 * mu_1 * mu_2],
            [0.7 * mu_1 * mu_2, (0.3 * mu_2) ** 2],
        ]
    )
    Y_1, Y_2 = np.random.multivariate_normal(mean, cov, size=N).T
    plot_scatter(Y_1, Y_2)
