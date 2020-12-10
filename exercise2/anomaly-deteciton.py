# 授業で説明したシミュレーションの方法より、ネットワークトラヒックを発生し、異常検出ができるかどうか確認しなさい
# 主成分分析による異常検出
from pprint import pprint

import numpy as np
from sklearn.decomposition import PCA

np.random.seed(1)

# 1. ネットワークトポロジーを設定
def calc_cov(i, j, var, hops, delta):
    if i == j:
        return var
    else:
        var *= np.exp(-hops[i][j] / delta)
        return var


N_node = 11
N_link = 10
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
delta = 0.1  # 0.1, 5.0
cov = np.zeros([len(hops), len(hops)])
r = 5

for i in range(len(hops)):
    for j in range(len(hops[0])):
        var = calc_cov(i, j, sigma ** 2, hops, delta)
        cov[i][j] = var
print(f'cov: {cov.shape}')

# 2. 時系列データを生成
X = np.random.multivariate_normal([mu] * cov.shape[0], cov, 100)
print(f"X: {X.shape}")

# 3. データから時間平均を引き、r つの固有ベクトルで部分空間にマップする
def map_ortho_comp(R, x_t):
    """直交補空間に射影
    :param R: (N, r), PCA により得られた固有ベクトル
    :param x_t: (N), 時間 t におけるトラヒック
    :return: y_tilde 射影された点
    """
    I = np.eye(R.shape[0])
    y_tilde = (I - R @ R.T) @ x_t
    return y_tilde


X_mean = np.average(X, 0)
pca = PCA()
pca.fit(X - X_mean)
R = pca.components_[:r, :].T
y_tilde = np.array([map_ortho_comp(R, x_t) for x_t in (X - X_mean)])
print(f"y_tilde: {y_tilde.shape}")

# 4. 閾値を設定し、異常であるかどうか分類する
# 5. Precision, Recall などで評価
