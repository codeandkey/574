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

    return Dataset(samples, labels)

class Dataset:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
