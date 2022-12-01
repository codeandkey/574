from dtree import DecisionTree
from classifier import Classifier
from args import args

import math
import numpy as np

class RandomForest(Classifier):
    def __init__(self, features):
        super().__init__()
        self.count = args.trees
        self.trees = [DecisionTree(features) for _ in range(self.count)]

    def train(self, samples, labels):
        # Divide samples into collections for each dtree.
        ds_samples = []
        ds_labels = []

        batch = int(math.ceil(len(samples) / self.count))

        # Build batch samples (WITH replacement/repeat samples) for each dtree
        for i in range(self.count):
            indices = np.arange(0, len(samples))
            np.random.shuffle(indices)
            ds_samples.append(samples[indices[:batch]])
            ds_labels.append(labels[indices[:batch]])

        for dtree, samples, labels in zip(self.trees, ds_samples, ds_labels):
            dtree.train(samples, labels)

    def infer(self, samples):
        totals = np.sum([tree.infer(samples) for tree in self.trees], axis=0)
        return np.round(totals / self.count)
