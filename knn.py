import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris #ur lucky iris is built in to sklearn
from sklearn.neighbors import KNeighborsClassifier

'''
Project Description
- allows us to classify species of iris flowers
- characteristics of a flower: ['sepal length (cm)', 'sepal width (cm)', 
'petal length (cm)', 'petal width (cm)']
for more info on the dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set
'''

'''Data manipulation'''
iris = load_iris()
print(iris.data)
#print(iris.feature_names) --> shows features
#print(iris.target_names) --> prints target names, 0 = setosa, 1 = versicolor, 2 = virginica
print(iris.keys())


'''Visualization'''
x_i = 0
y_i = 1

formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_i], iris.data[:, y_i], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.title('Iris Scatter Plot')
plt.xlabel(iris.feature_names[x_i])
plt.ylabel(iris.feature_names[y_i])
plt.show()
x = iris['data']
y = iris['target']

'''Training'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)

'''Accuracy'''
knn.fit(x_train, y_train)
print(knn.score(x_test, y_test))

'''Predictions'''
x_new_value = np.array([[5.0, 2.9, 1.0, 0.2]])
print(knn.predict(x_new_value)) # since the output of this is 0, the flower is setosa