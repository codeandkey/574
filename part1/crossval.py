import numpy as np
from dataset import Dataset
from args import args

def kfold(classifier, dataset, k=args.folds):
    """Performs K-fold cross validation for a classifier type."""

    sample_folds = np.array_split(dataset.samples, k)
    label_folds = np.array_split(dataset.labels, k)

    test_acc_total = 0
    test_prec_total = 0
    test_recall_total = 0
    test_f1_total = 0

    train_acc_total = 0
    train_prec_total = 0
    train_recall_total = 0
    train_f1_total = 0

    for i in range(k):
        test_samples = sample_folds[i]
        test_labels = label_folds[i]

        train_samples = np.concatenate([sample_folds[j] for j in range(k) if i != j])
        train_labels = np.concatenate([label_folds[j] for j in range(k) if i != j])

        train_dataset = Dataset(train_samples, train_labels, dataset.dmap)
        test_dataset = Dataset(test_samples, test_labels, dataset.dmap)

        c = classifier(dataset.features())
        c.train(train_samples, train_labels)

        train_acc, train_prec, train_recall, train_f1 = c.accuracy(train_samples, train_labels)
        test_acc, test_prec, test_recall, test_f1 = c.accuracy(test_samples, test_labels)

        train_acc_total += train_acc
        train_prec_total += train_prec
        train_recall_total += train_recall
        train_f1_total += train_f1

        test_acc_total += test_acc
        test_prec_total += test_prec
        test_recall_total += test_recall
        test_f1_total += test_f1

    train_acc_total /= k
    train_prec_total /= k
    train_recall_total /= k
    train_f1_total /= k

    test_acc_total /= k
    test_prec_total /= k
    test_recall_total /= k
    test_f1_total /= k

    return train_acc, train_prec, train_recall, train_f1, test_acc, test_prec, test_recall, test_f1

def loocv(classifier, dataset):
    return kfold(classifier, dataset, len(dataset.samples))
