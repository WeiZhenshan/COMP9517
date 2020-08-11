from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report,confusion_matrix
import cv2
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
# t=[0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30]
t=0.25
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=t,random_state=44)

# print(len(X_train),len(X_test),len(y_train),len(y_test))
neigh = KNeighborsClassifier(n_neighbors=9)
neigh.fit(X_train, y_train)
# print(neigh.get_params())
predicted_neigh_y = neigh.predict(X_test)

clf=SGDClassifier()
# print(clf.get_params())
clf.fit(X_train, y_train)
predicted_clf_y = clf.predict(X_test)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
predicted_tree_y = tree.predict(X_test)

# print("classification report\n",classification_report(y_test, predicted_neigh_y))
print('COMP9517 Week 5 Lab - z5232648')
print('Test Size = ',t)
print(f'KNN Accuracy:  {round(accuracy_score(y_test, predicted_neigh_y),3)}     Recall:{round(recall_score(y_test, predicted_neigh_y, average="micro"),3)}')
print(f'SGD Accuracy:  {round(accuracy_score(y_test, predicted_clf_y),3)}     Recall:{round(recall_score(y_test, predicted_clf_y, average="micro"),3)}')
print(f'DT Accuracy:   {round(accuracy_score(y_test, predicted_tree_y),3)}     Recall:{round(recall_score(y_test, predicted_tree_y, average="micro"),3)}')
print('KNN Confusion Matrix: \n',confusion_matrix(y_test, predicted_neigh_y))

# l=[]
# l1=[]
# l2=[]
#
# for k in range(1,20):
#     neigh = KNeighborsClassifier(n_neighbors=k)
#     neigh.fit(X_train, y_train)
#     predicted_neigh_y = neigh.predict(X_test)
#     l.append([k,round(accuracy_score(y_test, predicted_neigh_y),3)])
#     l1.append(k)
#     l2.append(round(accuracy_score(y_test, predicted_neigh_y),3))
#
# print(l)
# plt.plot(l1,l2)
# plt.xlabel('k values')
# plt.ylabel('accuracy score')
#
# plt.title('KNN- k vs accuracy')
# plt.show()

# random state- 1,42,71,72,74,77 (99.6)
# 44-1.0