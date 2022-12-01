from classifier import Classifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class DecisionTree(Classifier):
    def __init__(self, features):
        self.clf = DecisionTreeClassifier()

    def train(self, samples, __labels):
        labels = __labels * 2 - 1
        self.clf.fit(samples, labels)

    def infer(self, samples):
        predictions = self.clf.predict(samples)
        return (predictions > 0).astype(int)
