from sklearn.metrics import precision_score, recall_score, f1_score, \
                            accuracy_score
import numpy as np


class EmbeddingModelMetrics:
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.recall_at_k_values = {}
        self.rank_at_50_values = []

    def update(self, true_labels, predicted_labels):
        self.true_labels.extend(true_labels)
        self.predicted_labels.extend(predicted_labels)

        # Update recall at every 5
        for k in range(1, 16, 5):
            if k not in self.recall_at_k_values:
                self.recall_at_k_values[k] = []
            self.recall_at_k_values[k].append(self.calculate_recall_at_k(k))

        # Update rank at 50
        self.rank_at_50_values.append(self.calculate_rank_at_k(50))

    def calculate_precision(self):
        return precision_score(self.true_labels, self.predicted_labels)

    def calculate_recall(self):
        return recall_score(self.true_labels, self.predicted_labels)

    def calculate_f1_score(self):
        return f1_score(self.true_labels, self.predicted_labels)

    def calculate_accuracy(self):
        return accuracy_score(self.true_labels, self.predicted_labels)

    def calculate_recall_at_k(self, k):
        # Sort the predictions and select the top k
        top_k_indices = np.argsort(self.predicted_labels)[-k:]
        top_k_true_labels = np.array(self.true_labels)[top_k_indices]

        # Calculate
        recall_at_k = np.sum(top_k_true_labels) / np.sum(self.true_labels)
        return recall_at_k

    def calculate_rank_at_k(self, k):
        # Find the rank of the true label within the top k predictions
        true_label_index = np.where(np.array(self.true_labels) == 1)[0][0]
        top_k_indices = np.argsort(self.predicted_labels)[-k:]

        # Calculate
        ## 1 to start ranking from 1
        rank_at_k = np.where(top_k_indices == true_label_index)[0][
                        0] + 1
        return rank_at_k