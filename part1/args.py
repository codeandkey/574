import argparse

parser = argparse.ArgumentParser(description='574 final project')

parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='Learning rate',
)

parser.add_argument(
    '--iterations',
    type=int,
    default=1600,
    help='Training iterations',
)

parser.add_argument(
    '--dataset',
    default='dataset1.txt',
    help='Input dataset',
)

parser.add_argument(
    '--trees',
    default=10,
    help='Trees used in random forest',
    type=int
)

parser.add_argument(
    '--folds',
    default=10,
    help='Number of folds for K-fold',
)

parser.add_argument(
    '--reg',
    default=0.001,
    help='Regulariziation weight',
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

args = parser.parse_args()
