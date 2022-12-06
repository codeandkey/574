from classifier import Classifier
from sklearn.svm import SVC
from args import args

import numpy as np
import math

class SVM(Classifier):
    def __init__(self):
        self.clf = SVC(C=args.reg)

    def train(self, samples, labels):
        self.clf.fit(samples, labels * 2 - 1)

    def infer(self, samples):
        predictions = self.clf.predict(samples)
        return (predictions > 0).astype(int)
