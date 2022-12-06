from classifier import Classifier
from args import args
from sklearn.ensemble import RandomForestClassifier

class RandomForest(Classifier):
    def __init__(self):
        super().__init__()
        self.clf = RandomForestClassifier(n_estimators=args.trees, criterion=args.criterion)

    def train(self, samples, labels):
        self.clf.fit(samples, labels)

    def infer(self, samples):
        return self.clf.predict(samples)
