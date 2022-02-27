# coding:utf-8
import numpy as np

from sklearn.ensemble import AdaBoostClassifier

# 加载训练数据
X = np.array([[0, 1, 3], [0, 3, 1], [1, 2, 2], [1, 1, 3], [1, 2, 3], [0, 1, 2],
              [1, 1, 2], [1, 1, 1], [1, 3, 1], [0, 2, 1]])
y = np.array([-1, -1, -1, -1, -1, -1, 1, 1, -1, -1])
clf = AdaBoostClassifier()
clf.fit(X, y)
y_predict = clf.predict(X)
score = clf.score(X, y)
print("原始输出:", y)
print("预测输出:", y_predict)
print("预测正确率：{:.2%}".format(score))
