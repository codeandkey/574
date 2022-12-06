from crossval import evaluate
from svm import SVM
from dtree import DecisionTree
from dataset import Dataset
from forest import RandomForest
from args import args

import dataset
import argparse

if __name__ == '__main__':
    d = dataset.load(args.dataset)

    if args.classifier == 'svm':
        ctype = SVM
    elif args.classifier == 'dtree':
        ctype = DecisionTree
    elif args.classifier == 'forest':
        ctype = RandomForest
    else:
        raise RuntimeError('invalid classifier')

    train_info, test_info = evaluate(ctype, d)

    print(f'acc {train_info.accuracy:.3f} {test_info.accuracy:.3f}')
    print(f'prec {train_info.precision:.3f} {test_info.precision:.3f}')
    print(f'recall {train_info.recall:.3f} {test_info.recall:.3f}')
    print(f'f1 {train_info.f1:.3f} {test_info.f1:.3f}')
