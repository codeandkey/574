class Classifier:
    def train(self, dataset):
        """Trains this classifier on some training data."""
        raise RuntimeError('invalid classifier')

    def infer(self, samples):
        """Classifies a collection of samples."""
        raise RuntimeError('invalid classifier')

    def accuracy(self, samples, labels):
        """Tests the accuracy of the classifier on a collection of samples."""
        pred = self.infer(samples)

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

        prec = tp / (tp + fp + .0001)
        recall = tp / (tp + fn + .0001)

        return ((tp + tn) / (tp + tn + fp + fn)), prec, recall, 2 * (prec * recall) / (prec + recall + 0.0001)
