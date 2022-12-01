from crossval import kfold, loocv
from svm import SVM
from dtree import DecisionTree
from dataset import Dataset
from forest import RandomForest
from args import args

import dataset
import argparse

if __name__ == '__main__':
    d = dataset.load(args.dataset)

    ctype = None

    if args.classifier == 'svm':
        ctype = SVM
    elif args.classifier == 'dtree':
        ctype = DecisionTree
    elif args.classifier == 'forest':
        ctype = RandomForest
    else:
        raise RuntimeError('invalid classifier type')

    if args.eval == 'kfold':
        results = kfold(ctype, d)
    elif args.eval == 'loocv':
        results = loocv(ctype, d)

    train_acc, train_prec, train_recall, train_f1, test_acc, test_prec, test_recall, test_f1 = results

    print('train:')
    print(f'\tacc {train_acc:.3f}')
    print(f'\tprec {train_prec:.3f}')
    print(f'\trecall {train_recall:.3f}')
    print(f'\tf1 {train_f1:.3f}')

    print('test:')
    print(f'\tacc {test_acc:.3f}')
    print(f'\tprec {test_prec:.3f}')
    print(f'\trecall {test_recall:.3f}')
    print(f'\tf1 {test_f1:.3f}')
