import argparse

parser = argparse.ArgumentParser(description='574 final project')

parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='Learning rate',
)

parser.add_argument(
    '--reg',
    default=1,
    type=float,
    help='SVC regularization constant',
)

parser.add_argument(
    '--dataset',
    default='dataset1.txt',
    help='Input dataset',
)

parser.add_argument(
    '--trees',
    default=100,
    help='Trees used in random forest',
    type=int
)

parser.add_argument(
    '--folds',
    default=10,
    help='Number of folds for K-fold',
)

parser.add_argument(
    '--classifier',
    default='svm',
    help='Classification method (svm, dtree)',
)

parser.add_argument(
    '--eval',
    default='kfold',
    choices=['kfold', 'loocv'],
    help='Evaluation method',
    type=str
)

parser.add_argument(
    '--criterion',
    default='gini',
    choices=['gini', 'log_loss', 'entropy'],
    help='Decision tree splitting strategy',
    type=str
)

args = parser.parse_args()
