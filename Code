from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris = datasets.load_iris()
x = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21, stratify=y)
print(type(X_train))
print(type(X_test))

ks = range(1, 4)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(X_train)
    model.fit(X_test)
    inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o')
plt.xlabel('Number of Clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
