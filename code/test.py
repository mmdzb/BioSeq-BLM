import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from pytorch_pretrained_bert import BertTokenizer
import torch.nn as nn
import torch


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# with open("vocabulary.txt", "w", encoding='utf-8') as f:
#     for value in tokenizer.vocab.keys():
#         f.write(str(value) + '\n')



# iris = datasets.load_iris()
# iris_X = iris['data']
# iris_Y = iris['target']
#
# # print(iris_X[:2, :])
# # print(iris_Y[:2])
#
# X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=0.3)
#
# # print(Y_test)
#
# # knn = KNeighborsClassifier()
# # knn.fit(X_train, Y_train)
# # print(knn.predict(X_test))
# # print(Y_test)
#
# boston = datasets.load_boston()
# data_X = boston['data']
# data_y = boston['target']
#
# model = LinearRegression()
# model.fit(data_X, data_y)
# print(model.predict(data_X[:4, :]))
# print(data_y[:4])
# print(model.coef_)
# print(model.intercept_)

loss = nn.BCEWithLogitsLoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
output = loss(input, target)
print(output)