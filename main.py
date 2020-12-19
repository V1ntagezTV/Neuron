import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

#author InsurKun
#last edit: Okt. 15.09.2020, 12:46

data = read_csv('winequality-white.csv', delimiter=";")

X = data.iloc[:, :-1].values
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)
LinClassifier = LogisticRegression(max_iter=1000).fit(X_train, y_train)
lin_pred = LinClassifier.predict(X_test)

svcModel = svm.SVC()
svcModel.fit(X_train, y_train)
svc_pred = svcModel.predict(X_test)

print(accuracy_score(svc_pred, y_test))
print(accuracy_score(lin_pred, y_test))
print(confusion_matrix(svc_pred, y_test))
print(classification_report(lin_pred, y_test, zero_division=1))

plt.scatter(svc_pred, y_test, c='blue')
plt.legend(loc="upper right")
plt.xlabel("Quality")
plt.ylabel("Range")
plt.legend()
plt.show()


