from classifier import Classifier
from sklearn.tree import DecisionTreeClassifier
from args import args

class DecisionTree(Classifier):
    def __init__(self):
        self.clf = DecisionTreeClassifier(criterion=args.criterion)

    def train(self, samples, __labels):
        labels = __labels * 2 - 1
        self.clf.fit(samples, labels)

    def infer(self, samples):
        predictions = self.clf.predict(samples)
        return (predictions > 0).astype(int)
