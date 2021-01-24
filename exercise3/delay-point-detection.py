# 巨大な遅延時間を発生しているノードを発見する
import matplotlib.pyplot as plt
import numpy as np

# ネットワークの設定
print("\n1. define network...")
N = 21
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
max_eig_AtA = np.linalg.eig(A.T @ A)[0][0].real
c = int(max_eig_AtA) + 1
print(f" - routes: {routes}")
print(f" - A: {A}")
print(f" - max(eig(AtA)): {max_eig_AtA}")
print(f" - parameter c: {c}")
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
y = np.array(y)
print(f" - y: {y.shape}")
print("done")

print("\n3. initialize delay matrix...")
X = A.T @ np.average(y, axis=1)
print(f" - X: {X}")
print("done")

print("\n4. update...")
lam = 1


def update(X, y, lam, c):
    return soft_threshold(X - (1 / c) * A.T @ (A @ X - y), lam / c)


def soft_threshold(X, gamma):
    return np.sign(X) * np.maximum(0, np.abs(X) - gamma)


def evaluate(X, y, A, lam):
    loss = 0
    for i in range(y.shape[1]):
        loss += np.linalg.norm(A @ X - y[:, i]) ** 2 + lam * np.sum(np.abs(X))
    return loss / y.shape[1]


def visualize(X):
    plt.xlabel("nodes")
    plt.ylabel("delay")
    plt.bar(range(1, len(X) + 1), X)
    plt.xticks(range(1, len(X) + 1), range(1, len(X) + 1))
    plt.show()


for epoch in range(100):
    print(f" - epoch: {epoch}")
    for i in range(y.shape[1]):
        X_update = update(X, y[:, i], lam, c)
        X = X_update
    print(f" - evaluate: {evaluate(X, y, A, lam)}")

print(f" - X: {X}")
visualize(X)

print("done")
