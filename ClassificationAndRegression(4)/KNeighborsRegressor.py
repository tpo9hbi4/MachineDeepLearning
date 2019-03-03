import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
#Сгенерируем данные
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
print(X)
T = np.linspace(0, 5, 500)[:, np.newaxis]
print(T)
y = np.sin(X).ravel()
print(y)


y[::5] += 1 * (0.5 - np.random.rand(8))


n_neighbors = 5


knn = neighbors.KNeighborsRegressor(n_neighbors)
y_ = knn.fit(X, y).predict(T)

plt.subplot(2, 1, 1)
plt.scatter(X, y, c='k', label='data')
plt.plot(T, y_, c='g', label='prediction')
plt.axis('tight')
plt.legend()
plt.tight_layout()
plt.show()
