import numpy as np

from typing import List


class EmbeddingModelMetrics:
    def __init__(self, true_labels: List[int], predicted_labels: List[List[int]]):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels  # List

    def calculate_recall(self, k):
        predicted_labels_k = [labels[:k] for labels in self.predicted_labels]
        predicted_accuracy = [
            1 if self.true_labels[i] in predicted_labels_k[i] else 0
            for i in range(len(self.true_labels))
        ]
        return np.mean(predicted_accuracy)

    def calculate_rank(self):
        true_label_rank = [np.inf for _ in range(len(self.true_labels))]
        for i, labels in enumerate(self.predicted_labels):
            if self.true_labels[i] in labels:
                true_label_rank[i] = (
                    np.where(np.array(labels) == self.true_labels[i])[0][0] + 1
                )

        filter_true_label_rank = [
            td_rank for td_rank in true_label_rank if td_rank < np.inf
        ]
        return (
            np.mean(filter_true_label_rank),
            np.mean([1.0 / rank for rank in true_label_rank]),
            np.mean([1.0 / (np.log2(rank + 1)) for rank in true_label_rank]),
        )
