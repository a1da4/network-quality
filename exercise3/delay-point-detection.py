# 巨大な遅延時間を発生しているノードを発見する
import numpy as np

# ネットワークの設定
print("\n1. define network...")
N = 21
hops = [
    [0, 4, 4, 4, 4, 5, 5, 4, 5, 5, 4, 4, 2, 1, 3, 3, 4, 4, 3, 2, 3],
    [4, 0, 2, 4, 4, 5, 5, 4, 5, 5, 4, 4, 4, 3, 1, 3, 4, 4, 3, 2, 3],
    [4, 2, 0, 4, 4, 5, 5, 4, 5, 5, 4, 4, 4, 3, 1, 3, 4, 4, 3, 2, 3],
    [4, 4, 4, 0, 2, 5, 5, 4, 5, 5, 4, 4, 4, 3, 3, 1, 4, 4, 3, 2, 3],
    [4, 4, 4, 2, 0, 5, 5, 4, 5, 5, 4, 4, 4, 3, 3, 1, 4, 4, 3, 2, 3],
    [5, 5, 5, 5, 5, 0, 2, 3, 4, 4, 5, 5, 5, 4, 4, 4, 1, 3, 4, 3, 2],
    [5, 5, 5, 5, 5, 2, 0, 3, 4, 4, 5, 5, 5, 4, 4, 4, 1, 3, 4, 3, 2],
    [4, 4, 4, 4, 4, 3, 3, 0, 3, 3, 4, 4, 4, 3, 3, 3, 2, 2, 3, 2, 1],
    [5, 5, 5, 5, 5, 4, 4, 3, 0, 2, 5, 5, 5, 4, 4, 4, 3, 1, 4, 3, 2],
    [5, 5, 5, 5, 5, 4, 4, 3, 2, 0, 5, 5, 5, 4, 4, 4, 3, 1, 4, 3, 2],
    [4, 4, 4, 4, 4, 5, 5, 4, 5, 5, 0, 2, 4, 3, 3, 3, 4, 4, 1, 2, 3],
    [4, 4, 4, 4, 4, 5, 5, 4, 5, 5, 2, 0, 4, 3, 3, 3, 4, 4, 1, 2, 3],
    [2, 4, 4, 4, 4, 5, 5, 4, 5, 5, 4, 4, 0, 1, 3, 3, 4, 4, 3, 2, 3],
    [1, 3, 3, 3, 3, 4, 4, 3, 4, 4, 3, 3, 1, 0, 2, 2, 3, 3, 2, 1, 2],
    [3, 1, 1, 3, 3, 4, 4, 3, 4, 4, 3, 3, 3, 2, 0, 2, 3, 3, 2, 1, 2],
    [3, 3, 3, 1, 1, 4, 4, 3, 4, 4, 3, 3, 3, 2, 2, 0, 3, 3, 2, 1, 2],
    [4, 4, 4, 4, 4, 1, 1, 2, 3, 3, 4, 4, 4, 3, 3, 3, 0, 2, 3, 2, 1],
    [4, 4, 4, 4, 4, 3, 3, 2, 1, 1, 4, 4, 4, 3, 3, 3, 2, 0, 3, 2, 1],
    [3, 3, 3, 3, 3, 4, 4, 3, 4, 4, 1, 1, 3, 2, 2, 2, 3, 3, 0, 1, 2],
    [2, 2, 2, 2, 2, 3, 3, 2, 3, 3, 2, 2, 2, 1, 1, 1, 2, 2, 1, 0, 1],
    [3, 3, 3, 3, 3, 2, 2, 1, 2, 2, 3, 3, 3, 2, 2, 2, 1, 1, 2, 1, 0],
]
for i in range(N):
    for j in range(N):
        assert hops[i][j] == hops[j][i], f"DistanceDoesNotMatch {i}, {j}"

routes = [
    [0, 13, 19, 20, 17, 8],
    [1, 14, 19, 18, 11],
    [2, 14, 19, 20, 16, 6],
    [3, 15, 19, 13, 12],
    [4, 15, 19, 18, 10],
    [5, 16, 20, 7],
    [9, 17, 20, 19, 13, 12],
    [6, 16, 20, 17, 9],
    [7, 20, 19, 14, 2],
    [8, 17, 20, 19, 15, 4],
]
A = np.zeros([len(routes), N])
for i, route in enumerate(routes):
    for node in route:
        A[i][node] += 1
print(f" - routes: {routes}")
print(f" - A: {A}")
print("done")

# データの読み込み
print("\n2. load delay dataset...")
y = []
with open("delay.csv") as fp:
    for line in fp:
        delays = "".join(line.strip().split())[:-1]
        delays = delays.split(",")
        delays = [float(delay) for delay in delays]
        y.append(delays)
        #print(delays)
        #print(len(delays))
y = np.array(y)
print(f" - y: {y.shape}")
print("done")

# 各ノードの遅延時間 X の初期値は、ノードの hop に応じて決めた方が良さそう
print("\n3. initialize delay matrix...")
def calc_cov(i, j, var, hops, delta):
    if i == j:
        return var
    else:
        var *= np.exp(-hops[i][j] / delta)
        return var

# データを元に平均を出したい
#mu = np.average(A)
mu = np.average(y)
print(f" - mu: {mu}")
c = 2.0
sigma = c * mu
delta = 5.0
cov = np.zeros([N, N])

for i in range(N):
    for j in range(N):
        var = calc_cov(i, j, sigma ** 2, hops, delta)
        cov[i][j] = var

#print(cov)
mean = np.array([mu for _ in range(N)])
X = np.random.multivariate_normal(mean, cov)
print(f" - X: {X}")
print("done")
