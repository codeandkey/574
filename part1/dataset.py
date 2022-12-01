import numpy as np
import random

def load(path):
    samples = []
    dmap = {}

    def parse(word, feature):
        if word.replace('.', '').isnumeric():
            return float(word)

        if feature not in dmap:
            dmap[feature] = {}

        if word not in dmap[feature]:
            dmap[feature][word] = len(dmap[feature])

        return dmap[feature][word]

    with open(path, 'r') as f:
        for line in f:
            sample = [
                parse(word, feature) for feature, word
                in enumerate(line.split())
            ]

            samples.append(sample)

    samples = np.array(samples)
    np.random.shuffle(samples)

    labels = samples[:,-1]
    samples = samples[:,:-1]

    return Dataset(samples, labels, dmap)

class Dataset:
    def __init__(self, samples, labels, dmap):
        self.samples = samples
        self.labels = labels
        self.dmap = dmap

    def features(self):
        return self.samples.shape[1]

    def discrete(self, feature):
        return len(self.dmap[feature]) if feature in self.dmap else None

    # Only two filters..
    def filter(self, feature, threshold, comparator=None):
        samples = []
        labels = []

        if comparator:
            # Filter by continuous comparator
            if feature in self.dmap:
                raise RuntimeError('discrete features cannot use a comparator')
            
            for s, l in zip(self.samples, self.labels):
                if comparator(s[feature], threshold):
                    samples.append(s)
                    labels.append(l)
        else:
            # Filter by discrete matching, treat threshold as index
            if feature not in self.dmap:
                raise RuntimeError('continuous features require a comparator')

            target = threshold / len(self.dmap[feature])

            for s, l in zip(self.samples, self.labels):
                if s[feature] == target:
                    samples.append(s)
                    labels.append(l)

        return Dataset(np.array(samples), np.array(labels), self.dmap)
