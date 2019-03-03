#импортируем библиотеки
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#загрузим ириски
iris = load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state = 241)
#построим модель, используя деревья принятий решений.
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
#предскажем данные
y_new = clf.predict(X_test)
#увидим долю верных ответов
print(accuracy_score(y_test,y_new))
tree.export_graphviz(clf,out_file="treeClassification")
importances = clf.feature_importances_
#увидим наиболее важные признаки (см. рис.2.1)
print(importances)
