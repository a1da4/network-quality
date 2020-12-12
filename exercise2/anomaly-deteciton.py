# 授業で説明したシミュレーションの方法より、ネットワークトラヒックを発生し、異常検出ができるかどうか確認しなさい
# 主成分分析による異常検出
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(1)

# 1. ネットワークトポロジーを設定
N_node = 11
N_link = 10
N_data = 100
hops = [
    [0, 1, 1, 1, 2, 2, 2, 2, 3, 3],
    [1, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    [1, 1, 0, 1, 2, 2, 2, 2, 3, 3],
    [1, 1, 1, 0, 2, 2, 1, 1, 3, 3],
    [2, 1, 2, 2, 0, 1, 3, 3, 2, 2],
    [2, 1, 2, 2, 1, 0, 3, 3, 1, 1],
    [2, 2, 2, 1, 3, 3, 0, 1, 4, 4],
    [2, 2, 2, 1, 3, 3, 1, 0, 4, 4],
    [3, 2, 3, 3, 2, 1, 4, 4, 0, 1],
    [3, 2, 3, 3, 2, 1, 4, 4, 1, 0],
]
mu = 500
c = 2.0
sigma = c * mu
delta = 5  # 0.1, 5.0
cov = np.zeros([len(hops), len(hops)])
r = 5

abnormal_time = 70
abnormal_link = 0


def calc_cov(i, j, var, hops, delta):
    if i == j:
        return var
    else:
        var *= np.exp(-hops[i][j] / delta)
        return var


for i in range(len(hops)):
    for j in range(len(hops[0])):
        var = calc_cov(i, j, sigma ** 2, hops, delta)
        cov[i][j] = var

# 2. 時系列データを生成、スムージング
def plot_data(X, title):
    """データをプロットする
    :param X: データ
    """
    plt.xlabel("time")
    plt.ylabel("traffic")
    plt.plot(range(X.shape[0]), X.T[0], label=title[:-4])
    plt.legend()
    plt.savefig(title)
    plt.close()


X = np.random.multivariate_normal([mu] * cov.shape[0], cov, N_data)
X_mean = np.average(X, 0)
plot_data(X - X_mean, title="link-1_raw-mean.png")


def ewma_smoothing(X, alpha=0.03):
    """指数加重移動平均
    :param X: (Time, N_link), データ
    :param alpha: 平滑化パラメータ
    :return X_smoothed: 平滑化されたデータ
    """
    X[0] *= alpha
    for t in range(1, X.shape[0]):
        X[t] = (1 - alpha) * X[t - 1] + alpha * X[t]
    return X


X_smoothed = ewma_smoothing(X)
plot_data(X_smoothed, title="link-1_smoothed.png")

# 3. データから時間平均を引き、r つの固有ベクトルで部分空間にマップする
#    異常なトラフィックもこの時に生成
def map_ortho_comp(R, x_t):
    """直交補空間に射影
    :param R: (N, r), PCA により得られた固有ベクトル
    :param x_t: (N), 時間 t におけるトラヒック
    :return: y_tilde 射影された点
    """
    I = np.eye(R.shape[0])
    y_tilde = (I - R @ R.T) @ x_t
    return y_tilde


pca = PCA()
pca.fit(X - X_mean)
X[abnormal_time][abnormal_link] += 700
R = pca.components_[:r, :].T
y_tilde = np.array([map_ortho_comp(R, x_t) for x_t in (X - X_mean)])
plot_data(y_tilde, title="link-1_ytilde.png")

# 4. 閾値を設定し、異常であるかどうか分類する
y_th = 300
for t in range(y_tilde.shape[0]):
    for i in range(y_tilde.shape[1]):
        if abs(y_tilde[t][i]) >= y_th:
            print(f"Detected! time-{t}, link-{i}")
