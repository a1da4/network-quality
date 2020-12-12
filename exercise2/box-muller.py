# box-muller 法で平均0、分散1の正規分布に従う確率変数 X1, X2 の乱数を生成し、1000組描画
import matplotlib.pyplot as plt
import numpy as np


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
        w = x_1 ** 2 + x_2 ** 2
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


if __name__ == "__main__":
    np.random.seed(1)
    mu_1 = 0
    sigma_1 = 1
    mu_2 = 0
    sigma_2 = 1
    X_1, X_2 = box_muller(mu_1, sigma_1, mu_2, sigma_2)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(X_1, X_2)
    plt.savefig("random-numbers_from_box-muller.png")
