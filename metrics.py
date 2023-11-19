from sklearn.metrics import precision_score, recall_score, f1_score, \
                            accuracy_score


class EmbeddingModelMetrics:
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels

    def calculate_precision(self):
        return precision_score(self.true_labels, self.predicted_labels, average='weighted')

    def calculate_recall(self):
        return recall_score(self.true_labels, self.predicted_labels, average='weighted')

    def calculate_f1_score(self):
        return f1_score(self.true_labels, self.predicted_labels, average='weighted')

    def calculate_accuracy(self):
        return accuracy_score(self.true_labels, self.predicted_labels)