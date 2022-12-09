import numpy as np
import random

def load(path):
    samples = []
    dmap = {}

    def parse(word, feature):
        """Parse continuous or discrete attributes."""

        # Check if continuous
        if word.replace('.', '').isnumeric():
            return float(word)

        # If not continuous, check if a discrete map exists for this feature
        if feature not in dmap:
            dmap[feature] = {}

        # Assign a new discrete value to this word
        if word not in dmap[feature]:
            dmap[feature][word] = len(dmap[feature])

        # Return discrete value for this word
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

    # Separate last column into labels
    labels = samples[:,-1]
    samples = samples[:,:-1]

    return Dataset(samples, labels)

class Dataset:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
