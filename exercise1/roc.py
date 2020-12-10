import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt


# \integral_{-inf}^{upper} normal(mu, sigma^2) dx
def s(upper, mu, sigma):
    value = 0.5 * (1 + erf((upper - mu) / np.sqrt(2) / sigma))
    return value


np.random.seed(1)
mu_p = 300
sigma_p = 90
mu_n = 100
sigma_n = 30

x_th_seq = range(0, 1000, 10)
r_tp_seq = []
r_fp_seq = []

for x_th in x_th_seq:
    # print(x_th)
    r_fn = s(x_th, mu_p, sigma_p)
    r_tp = 1 - r_fn
    r_tn = s(x_th, mu_n, sigma_n)
    r_fp = 1 - r_tn

    r_tp_seq.append(r_tp)
    r_fp_seq.append(r_fp)

# plot
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(r_fp_seq, r_tp_seq, color="blue", label="ROC_theory")

# シミュレーションで x を生成
# x ~ qN(mu_p, sigma_p^2) + (1 - q)N(mu_n, sigma_n^2)
# 異常確率 q = 0.3 とする
q = 0.5
x_th_seq = range(0, 1000, 50)

r_tp_seq = []
r_fp_seq = []

for x_th in x_th_seq:
    # サンプルを10000回生成
    p_seq = np.random.rand(10000)
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for p in p_seq:
        if p > q:
            # 正常(negative)
            x = np.random.normal(mu_n, sigma_n)
            # 予測、答え合わせ
            if x < x_th:
                # 正常と予測(True negative)
                tn += 1
            else:
                # 異常と予測(False positive)
                fp += 1
        else:
            # 異常(positive)
            x = np.random.normal(mu_p, sigma_p)
            # 予測、答え合わせ
            if x >= x_th:
                # 異常と予測(True positive)
                tp += 1
            else:
                # 正常と予測(False negative)
                fn += 1
    r_tp = tp / (tp + fn + 1e-8)  # recall
    r_fn = 1 - r_tp
    r_fp = fp / (tn + fp + 1e-8)
    r_tn = 1 - r_fp

    r_tp_seq.append(r_tp)
    r_fp_seq.append(r_fp)

plt.scatter(r_fp_seq, r_tp_seq, color="blue", label="ROC_sampled")
plt.legend()
plt.show()
