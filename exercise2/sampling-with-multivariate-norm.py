# 2次の多変量正規分布に従う乱数を生成
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    np.random.seed(1)
    mu_1 = 100
    mu_2 = 200
    N = 1000
    mean = np.array([mu_1, mu_2])
    sigma_1 = 0.3 * mu_1
    sigma_2 = 0.3 * mu_2
    sigma_12 = 0.7 * sigma_1 * sigma_2
    cov = np.array(
        [
            [sigma_1 ** 2, sigma_12],
            [sigma_12, sigma_2 ** 2],
        ]
    )
    Y_1, Y_2 = np.random.multivariate_normal(mean, cov, size=N).T
    plt.xlabel("Y1")
    plt.ylabel("Y2")
    plt.scatter(Y_1, Y_2)
    plt.savefig("random-numbers_from_multivariate-norm.png")
