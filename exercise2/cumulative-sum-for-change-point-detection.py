# 変化点を持つ時系列データを発生し、CUSUMで変化点検出
# G(n), S(n) のグラフを描画
import matplotlib.pyplot as plt
import numpy as np

# 1. データ量、変化点、変化の前後における分布のパラメータ
np.random.seed(1)

N_data = 100
change_point = 50

mu_1 = 10
var_1 = 10

mu_2 = 15
var_2 = 22.5


# 2. CUSUM で用いる関数 G(n), S(n) の準備
def p_1(x, mu=mu_1, var=var_1):
    """正規分布1（変化前）における確率密度関数
    :param x: データ
    """
    p = np.exp(-((x - mu) ** 2) / var) / np.sqrt(2 * np.pi * var)
    return p


def p_2(x, mu=mu_2, var=var_2):
    """正規分布2（変化後）における確率密度関数
    :param x: データ
    """
    p = np.exp(-((x - mu) ** 2) / var) / np.sqrt(2 * np.pi * var)
    return p


def s(x):
    """与えられたデータにおける変化前・後の尤度比
    :param x: ある時点でのデータ X[t]
    """
    return np.log(p_2(x) / p_1(x))


def logl_cusum(n, X):
    """尤度比の累積和
    :param n: 累積和を行う時間の範囲
    :param X: 全時点のデータ
    :return S: 時刻0からnにおける累積和
    """
    S = np.zeros(n)
    for t in range(n):
        x = X[t]
        S[t:] += s(x)
    return S


def max_logl(n, S):
    """最大尤度比
    :param n: 累積和を行う時間の範囲
    :param S: [N_data], 尤度比の累積和
    """
    S_n = S[n]
    if n > 0:
        S_min = np.min(S[:n])
    else:
        S_min = S[n]
    return S_n - S_min


def detect_change_point(S, G, h):
    """変化点（設定した閾値を超える点）を検出
    :param S: [N_data], 尤度比の累積和
    :param G: [N_data], 最大尤度比の集合
    :param h: int, 閾値（正の値）
    """
    t_stop_iter = 0
    T = len(G)
    for t in range(T):
        if G[t] > h:
            t_stop_iter = t
            break
    t_stop_iter = T
    kc = np.argmin(S[:t_stop_iter])
    return kc


# 3. メイン処理
## 3.1 データの生成
X = np.hstack(
    [
        np.random.normal(mu_1, np.sqrt(var_1), 50),
        np.random.normal(mu_2, np.sqrt(var_2), 50),
    ]
)

## 3.2 グラフの準備、データのプロット
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
plt.xlabel("time")
ax1.set_ylabel("x")
ax2.set_ylabel("S(n), G(n)")
ax1.vlines(
    change_point,
    np.max(X),
    np.min(X),
    colors="black",
    linestyle="solid",
    linewidth=3,
    label="change_point",
)
ax1.plot(range(N_data), X, label="X")

## 3.3 S(n), G(n) の計算
S = logl_cusum(N_data, X)
G = [max_logl(n, S) for n in range(N_data)]
ax2.plot(range(N_data), S, color="green", label="S(n)")
ax2.plot(range(N_data), G, color="orange", label="G(n)")

## 3.4 閾値 h を用いて、変化点 kc の予測
h = 20
kc = detect_change_point(S, G, h)
print(f"change point pred: {kc}")
ax2.hlines(
    h, 0, N_data, colors="red", linestyle="solid", linewidth=3, label="threshold"
)
ax1.vlines(
    kc,
    np.max(X),
    np.min(X),
    colors="black",
    linestyle="dashed",
    linewidth=3,
    label="change_point_pred",
)

handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()

ax1.legend(handler1 + handler2, label1 + label2)
plt.savefig("changepoint_by_cusum.png")
