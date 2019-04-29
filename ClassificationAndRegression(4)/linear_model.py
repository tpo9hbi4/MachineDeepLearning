import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Загрузим датасет
diabetes = datasets.load_diabetes()
print(diabetes)

# Выберем один признак
diabetes_X = diabetes.data[:, np.newaxis, 2]
print(diabetes_X)

# Выделим обучающие и тестовые признаки
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Выделим обучающие и тестовые ответы
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Создаём объект линейной регрессии
regr = linear_model.LinearRegression()
# обучим модель
regr.fit(diabetes_X_train, diabetes_y_train)
# Предскажем значение
diabetes_y_pred = regr.predict(diabetes_X_test)
# Выведем коэффициенты регрессии
print('Coefficients: \n', regr.coef_)
# Метрики оценки
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
# Изобразим точки и аппроксимирующую прямую (рис. 3.3)
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.xlabel('$data$')
plt.ylabel('$target$')
plt.show()
