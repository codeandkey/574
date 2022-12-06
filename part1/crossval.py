import numpy as np
from dataset import Dataset
from args import args
from classifier import AccuracyInfo
from sklearn.model_selection import KFold, LeaveOneOut
import math

def evaluate(classifier, dataset):
    """Evaluates the accuracy of a classifier using K-fold cross validation."""

    train_info = AccuracyInfo()
    test_info = AccuracyInfo()

    if args.eval == 'kfold':
        folder = KFold()
    elif args.eval == 'loocv':
        folder = LeaveOneOut()
    else:
        raise RuntimeError('invalid evaluator')

    for train_index, test_index in folder.split(dataset.samples):
        c = classifier()
        c.train(dataset.samples[train_index], dataset.labels[train_index])

        c.accuracy(dataset.samples[train_index],
                   dataset.labels[train_index],
                   train_info)

        c.accuracy(dataset.samples[test_index],
                   dataset.labels[test_index],
                   test_info)

    return train_info, test_info
