import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
sns.set(style='white')
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

# Загрузим датасет с данными цветов ирисов
iris = datasets.load_iris()
# Выделим матрицу признаков цветов (4 штуки) и столбец ответов, содержащий 3 класса ирисов -'Setosa' с кодом “0”, 'Versicolour' – “1” и 'Virginica' – “2”
X = iris.data
y = iris.target
#Напечатаем датасет ирисов и класс по каждому из них см.рис.2.1
print(X)
print(y)
# Настройка 3D сцены
fig = plt.figure(1, figsize=(6, 5))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

# Отрисуем точки разными цветами по трем первым признакам ирисов см. рис.2.2
y_clr = np.choose(y, ["blue","green","red"])
color = ("red","green","blue")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_clr, cmap=plt.cm.spectral)
plt.show()
#Загрузим пакет с методом PCA 
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
#Выполним центровку данных, путем вычитания среднего по столбцам
X_centered = X - X.mean(axis=0)
X_pca = pca.fit_transform(X_centered)
# размерность изменилась до 2D
# Нарисуем получившиеся точки в пространстве 2D см. рис. 2.3.
plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'bo', label='Setosa')
plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], 'go', label='Versicolour')
plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], 'ro', label='Virginica')
plt.legend(loc=0);
plt.show()
digits = datasets.load_digits()
X = digits.data
y = digits.target
digits = datasets.load_digits()
X = digits.data
y = digits.target
#Выведем матрицу с объектами и признаками рис.2.4
print(X)
print(y)
# Свернём признаковое описание в матрицу интенсивностей 8x8 и изобразим цифры
# см. рис. 2.5
plt.figure(figsize=(16, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i,:].reshape([8,8]));
plt.show()
pca = decomposition.PCA(n_components=2)
X_reduced = pca.fit_transform(X)
#Отобразим элементы датасета в новом 2-х мерном пространстве, раскрасив их в соответствии со значением y (см. рис. 2.6)
plt.figure(figsize=(12,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.show()
pca = decomposition.PCA().fit(X)
plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 63)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(21, c='b')
plt.axhline(0.9, c='r')
plt.show();
