from classifier import Classifier
from sklearn.svm import SVC

import numpy as np

class SVM(Classifier):
    def __init__(self, features):
        self.clf = SVC()

    def train(self, samples, __labels):
        # Convert labels from (0, 1) to (-1, 1)
        labels = __labels * 2 - 1

        self.clf.fit(samples, labels)

    def infer(self, samples):
        predictions = self.clf.predict(samples)
        return (predictions > 0).astype(int)
