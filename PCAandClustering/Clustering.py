#hierarchy.linkage
#Импортируем библиотеку scipy метод hierarchy для кластеризации и метод pdist для расчёта попарных расстояний
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

# Самостоятельно набросаем данные для кластеризации

X = np.zeros((150, 2))

np.random.seed(seed=42)
X[:50, 0] = np.random.normal(loc=0.0, scale=.3, size=50)
X[:50, 1] = np.random.normal(loc=0.0, scale=.3, size=50)

X[50:100, 0] = np.random.normal(loc=2.0, scale=.5, size=50)
X[50:100, 1] = np.random.normal(loc=-1.0, scale=.2, size=50)

X[100:150, 0] = np.random.normal(loc=-1.0, scale=.2, size=50)
X[100:150, 1] = np.random.normal(loc=2.0, scale=.5, size=50)

distance_mat = pdist(X) 
# pdist посчитает нам верхний треугольник матрицы попарных Евклидовых 
# расстояний

Z = hierarchy.linkage(distance_mat, 'single') 
# linkage — реализация агломеративного алгоритма (см. рис. 2.9)
plt.figure(figsize=(10, 5))
dn = hierarchy.dendrogram(Z, color_threshold=0.5)
plt.show()
#Kmeans
# Начнём с того, что насыплем на плоскость три кластера точек (рис. 2.10)
Z = np.zeros((150, 2))

np.random.seed(seed=42)
Z[:50, 0] = np.random.normal(loc=0.0, scale=.3, size=50)
Z[:50, 1] = np.random.normal(loc=0.0, scale=.3, size=50)

Z[50:100, 0] = np.random.normal(loc=2.0, scale=.5, size=50)
Z[50:100, 1] = np.random.normal(loc=-1.0, scale=.2, size=50)

Z[100:150, 0] = np.random.normal(loc=-1.0, scale=.2, size=50)
Z[100:150, 1] = np.random.normal(loc=2.0, scale=.5, size=50)

plt.figure(figsize=(5, 5))
plt.plot(Z[:, 0], Z[:, 1], 'bo');
plt.show()

from sklearn.cluster import KMeans

inertia = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(Z)
    inertia.append(np.sqrt(kmeans.inertia_))

plt.plot(range(1, 8), inertia, marker='s');
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$');
plt.show()
#строим модель
kmeans = KMeans(n_clusters=3)
kmeans.fit(Z)
#проставленные метки алгоритмом
dataTrainY=kmeans.labels_
# Отображаем полученные кластеры на плоскости (рис. 2.12)
plt.plot(Z[dataTrainY==0,0],Z[dataTrainY==0,1], 'bo', label='class1')
plt.plot(Z[dataTrainY==1,0],Z[dataTrainY==1,1], 'go', label='class2')
plt.plot(Z[dataTrainY==2,0],Z[dataTrainY==2,1], 'ro', label='class3')
plt.legend(loc=0)
plt.show()
