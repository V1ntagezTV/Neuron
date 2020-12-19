import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tests.test_multiclass import datasets

# author InsurKun
# last edit: Okt. 17.09.2020, 16:22

iris_dataset = datasets.load_iris()

X_train, X_test, y_train, y_test = \
    train_test_split(iris_dataset.data, iris_dataset.target, test_size=0.20, random_state=27)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
clf = Perceptron(max_iter=100, n_jobs=-1, random_state=8)
clf.fit(X_train_std, y_train)
ypred = clf.predict(X_test_std)

first = 0
second = 0
third = 0
print(ypred)
for num in ypred:
    if num == 0:
        first+=1
    elif num == 1:
        second+=1
    else:
        third+=1

plt.bar([0,1,2], [first, second, third], label="Perceptrone")
plt.legend()
plt.xlabel('classification')
plt.ylabel('count')
plt.title('Таблица предсказанных данных!')
plt.show()

print(accuracy_score(ypred, y_test))
print(confusion_matrix(ypred, y_test))
print(classification_report(ypred, y_test, zero_division=1))