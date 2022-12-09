class AccuracyInfo:
    def __init__(self):
        self.accuracy_list = []
        self.precision_list = []
        self.recall_list = []
        self.f1_list = []

    def __getattr__(self, attr):
        """Computes an attribute average, or returns 0 if no records are
           present."""
        target = getattr(self, attr + '_list')

        if len(target) == 0:
            return 0

        return sum(target) / len(target)

class Classifier:
    def train(self, dataset):
        """Trains this classifier on some training data."""
        raise RuntimeError('invalid classifier')

    def infer(self, samples):
        """Classifies a collection of samples."""
        raise RuntimeError('invalid classifier')

    def accuracy(self, samples, labels, info):
        """Tests the accuracy of the classifier on a collection of samples. The
           resulting scores are stored in info: AccuracyInfo."""
        pred = self.infer(samples)

        # Count true positive, true negative, etc., and compute accuracy,
        # precision, recall and F1.

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i in range(len(labels)):
            if labels[i] == 1:
                if pred[i] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred[i] == 1:
                    fp += 1
                else:
                    tn += 1

        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0

        if tp + fp > 0:
            precision = tp / (tp + fp)
            info.precision_list.append(precision)

        if tp + fn > 0:
            recall = tp / (tp + fn)
            info.recall_list.append(recall)

        if tp + fp + tn + fn > 0:
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            info.accuracy_list.append(accuracy)

        if precision + recall > 0:
            f1 =  (precision * recall) / (precision + recall)
            info.f1_list.append(f1)
